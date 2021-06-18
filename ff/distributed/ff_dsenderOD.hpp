#ifndef FF_DSENDEROD_H
#define FF_DSENDEROD_H

#include <iostream>
#include <map>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <netdb.h>
#include <cmath>
#include <thread>

#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>


using namespace ff;

class ff_dsenderOD: public ff_minode_t<message_t> { 
private:
    size_t neos=0;
    int last_rr_socket = 0; //next destiation to send for round robin policy
    const int queueDim;
    std::vector<ff_endpoint> dest_endpoints;
    std::map<int, int> dest2Socket;
    std::map<int, unsigned int> sockCounters;
    std::map<int, int> sockets;
	int coreid;
    fd_set set, tmpset;
    int fdmax = -1;

    int receiveReachableDestinations(int sck){
        size_t sz;

        //while (readvn(sck, iov, 1) != sizeof(sz)) // wait untill a size is received!
        recv(sck, &sz, sizeof(sz), MSG_WAITALL);

        sz = be64toh(sz);

        std::cout << "Receiving routing table (" << sz << " bytes)" << std::endl;
        char* buff = new char [sz];
		assert(buff);

        if(readn(sck, buff, sz) < 0){
            error("Error reading from socket\n");
            delete [] buff;
            return -1;
        }

        dataBuffer dbuff(buff, sz, true);
        std::istream iss(&dbuff);
		cereal::PortableBinaryInputArchive iarchive(iss);
        std::vector<int> destinationsList;

        iarchive >> destinationsList;

        for (int d : destinationsList) dest2Socket[d] = sck;

        return 0;
    }
	
    int create_connect(const ff_endpoint& destination){
        int socketFD;

        #ifdef LOCAL
            socketFD = socket(AF_LOCAL, SOCK_STREAM, 0);
            if (socketFD < 0){
                error("\nError creating socket \n");
                return socketFD;
            }
            struct sockaddr_un serv_addr;
            memset(&serv_addr, '0', sizeof(serv_addr));
            serv_addr.sun_family = AF_LOCAL;

            strncpy(serv_addr.sun_path, destination.address.c_str(), destination.address.size()+1);

            if (connect(socketFD, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0){
                close(socketFD);
                return -1;
            }
        #endif

        #ifdef REMOTE
            struct addrinfo hints;
            struct addrinfo *result, *rp;

            memset(&hints, 0, sizeof(hints));
            hints.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
            hints.ai_socktype = SOCK_STREAM; /* Stream socket */
            hints.ai_flags = 0;
            hints.ai_protocol = IPPROTO_TCP;          /* Allow only TCP */

            // resolve the address 
            if (getaddrinfo(destination.address.c_str() , std::to_string(destination.port).c_str() , &hints, &result) != 0)
                return -1;

            // try to connect to a possible one of the resolution results
            for (rp = result; rp != NULL; rp = rp->ai_next) {
               socketFD = socket(rp->ai_family, rp->ai_socktype,
                            rp->ai_protocol);
               if (socketFD == -1)
                   continue;

               if (connect(socketFD, rp->ai_addr, rp->ai_addrlen) != -1)
                   break;                  /* Success */

               close(socketFD);
           }

           if (rp == NULL)            /* No address succeeded */
               return -1;
        #endif

        // receive the reachable destination from this sockets
        receiveReachableDestinations(socketFD);


        return socketFD;
    }

    int tryConnect(const ff_endpoint &destination){
        int fd, retries = 0;

        while((fd = this->create_connect(destination)) < 0 && ++retries < MAX_RETRIES)
            if (retries < AGGRESSIVE_TRESHOLD)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                //std::this_thread::sleep_for(std::chrono::milliseconds((long)std::pow(2, retries - AGGRESSIVE_TRESHOLD)));

        return fd;
    }

    int sendToSck(int sck, message_t* task){
        //std::cout << "received something from " << task->sender << " directed to " << task->chid << std::endl;
        task->sender = htonl(task->sender);
        task->chid = htonl(task->chid);

        size_t sz = htobe64(task->data.getLen());
        struct iovec iov[4];
        iov[0].iov_base = &task->sender;
        iov[0].iov_len = sizeof(task->sender);
        iov[1].iov_base = &task->chid;
        iov[1].iov_len = sizeof(task->chid);
        iov[2].iov_base = &sz;
        iov[2].iov_len = sizeof(sz);
        iov[3].iov_base = task->data.getPtr();
        iov[3].iov_len = task->data.getLen();

        if (writevn(sck, iov, 4) < 0){
            error("Error writing on socket\n");
            return -1;
        }

        return 0;
    }

    
public:
    ff_dsenderOD(ff_endpoint dest_endpoint, int queueDim = 1, int coreid=-1)
		: queueDim(queueDim), coreid(coreid) {
        this->dest_endpoints.push_back(std::move(dest_endpoint));
    }

    ff_dsenderOD( std::vector<ff_endpoint> dest_endpoints_, int queueDim = 1, int coreid=-1)
		: queueDim(queueDim), dest_endpoints(std::move(dest_endpoints_)),coreid(coreid) {}

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
		
        FD_ZERO(&set);
        FD_ZERO(&tmpset);

        for(size_t i=0; i < this->dest_endpoints.size(); i++){
            if ((sockets[i] = tryConnect(this->dest_endpoints[i])) <= 0 ) return -1;
            
            // execute the following block only if the scheduling is onDemand
            
            sockCounters[sockets[i]] = queueDim;
            FD_SET(sockets[i], &set);
            if (sockets[i] > fdmax)
                fdmax = sockets[i];
            
        }

        return 0;
    }

    int waitAckFrom(int sck){
        while (sockCounters[sck] == 0){
            for (int i = 0; i < this->sockets.size(); ++i){
                int r; ack_t a;
                if ((r = recvnnb(sockets[i], reinterpret_cast<char*>(&a), sizeof(ack_t))) != sizeof(ack_t)){
                    if (errno == EWOULDBLOCK){
                        assert(r == -1);
                        continue;
                    }
                    perror("readn ack");
                    return -1;
                } else 
                    //printf("received ACK from conn %d\n", i);
                    sockCounters[sockets[i]]++;
                
            }

            if (sockCounters[sck] == 0){
                tmpset = set;
                if (select(fdmax + 1, &tmpset, NULL, NULL, NULL) == -1){
                    perror("select");
                    return -1;
                }
            }
        }
        return 1;
    }

    bool isOneReady(){
        for(const auto& pair : sockCounters)
            if (pair.second > 0) return true;
        return false;
    }

    int getNextReady(){
        for(int i = 0; i < this->sockets.size(); i++){
            int actualSocket = (last_rr_socket + 1 + i) % this->sockets.size();
            int sck = sockets[actualSocket];
            if (sockCounters[sck] > 0) {
                last_rr_socket = actualSocket;
                return sck;
            }
        }

        // devo ricevere almeno un ack da qualcuno
        while(!isOneReady()){

            // se non ho ricevuto nessun ack mi sospendo finche non ricevo qualcosa
            tmpset = set;
            if (select(fdmax + 1, &tmpset, NULL, NULL, NULL) == -1){
                perror("select");
                return -1;
            } 

            for (int i = 0; i < this->sockets.size(); ++i){
                int r; ack_t a;
                int sck = sockets[i];
                if ((r = recvnnb(sck, reinterpret_cast<char*>(&a), sizeof(ack_t))) != sizeof(ack_t)){
                    if (errno == EWOULDBLOCK){
                        assert(r == -1);
                        continue;
                    }
                    perror("readn ack");
                    return -1;
                } else {
                    //printf("received ACK from conn %d\n", i);
                    
                    sockCounters[sck]++;
                    last_rr_socket = i;
                    return sck;
                }
            } 
        }
    }


    void svc_end() {
        // close the socket not matter if local or remote
        for(size_t i=0; i < this->sockets.size(); i++)
            close(sockets[i]);    
    }
    message_t *svc(message_t* task) {
        int sck;
        if (task->chid != -1){
            sck = dest2Socket[task->chid];
            if (sockCounters[sck] == 0 && waitAckFrom(sck) == -1){ // blocking call if scheduling is ondemand
                    error("Error waiting Ack from....\n");
                    delete task; return this->GO_ON;
            }
        } else 
            sck = getNextReady(); // blocking call if scheduling is ondemand
    
        sendToSck(sck, task);

        // update the counters
        sockCounters[sck]--;

        delete task;
        return this->GO_ON;
    }

     void eosnotify(ssize_t) {
	    if (++neos >= this->get_num_inchannels()){
            message_t * E_O_S = new message_t;
            E_O_S->chid = 0;
            E_O_S->sender = 0;
            for(const auto& pair : sockets)
                sendToSck(pair.second, E_O_S);

            delete E_O_S;
        }
    }


};

#endif
