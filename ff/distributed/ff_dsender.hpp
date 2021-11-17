#ifndef FF_DSENDER_H
#define FF_DSENDER_H

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

class ff_dsender: public ff_minode_t<message_t> { 
protected:
    int neos=0;
    int Ineos=0;
    int next_rr_destination = 0; //next destiation to send for round robin policy
    std::vector<ff_endpoint> dest_endpoints;
    std::map<int, int> dest2Socket;
    std::unordered_map<ConnectionType, std::vector<int>> type2sck;
    std::vector<int> sockets;
	int internalGateways;
    int coreid;

    int receiveReachableDestinations(int sck){
       
        size_t sz;
        ssize_t r;

        if ((r=readn(sck, (char*)&sz, sizeof(sz)))!=sizeof(sz)) {
            if (r==0)
                error("Error unexpected connection closed by receiver\n");
            else			
                error("Error reading size (errno=%d)");
            return -1;
        }
	
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
		   free(result);
			
           if (rp == NULL)            /* No address succeeded */
               return -1;
        #endif

        // store the connection type
        type2sck[destination.typ].push_back(socketFD);

        if (writen(socketFD, (char*) &destination.typ, sizeof(destination.typ)) < 0){
            error("Error sending the connection type during handshaking!");
            return -1;
        }

        // receive the reachable destination from this sockets
        if (receiveReachableDestinations(socketFD) < 0)
            return -1;


        return socketFD;
    }

    int tryConnect(const ff_endpoint &destination){
        int fd = -1, retries = 0;

        while((fd = this->create_connect(destination)) < 0 && ++retries < MAX_RETRIES)
            if (retries < AGGRESSIVE_TRESHOLD)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                //std::this_thread::sleep_for(std::chrono::milliseconds((long)std::pow(2, retries - AGGRESSIVE_TRESHOLD)));

        return fd;
    }

    int sendToSck(int sck, message_t* task){
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
    ff_dsender(ff_endpoint dest_endpoint, int internalGateways = 0, int coreid=-1): internalGateways(internalGateways), coreid(coreid) {
        this->dest_endpoints.push_back(std::move(dest_endpoint));
    }

    ff_dsender( std::vector<ff_endpoint> dest_endpoints_, int internalGateways = 0, int coreid=-1) : dest_endpoints(std::move(dest_endpoints_)), internalGateways(internalGateways), coreid(coreid) {}

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
		
        sockets.resize(this->dest_endpoints.size());
        for(size_t i=0; i < this->dest_endpoints.size(); i++)
            if ((sockets[i] = tryConnect(this->dest_endpoints[i])) <= 0 ) return -1;

        return 0;
    }

    void svc_end() {
        // close the socket not matter if local or remote
        for(size_t i=0; i < this->sockets.size(); i++)
            close(sockets[i]);
    }

    message_t *svc(message_t* task) {
        /* here i should send the task via socket */
        if (task->chid == -1){ // roundrobin over the destinations
            task->chid = next_rr_destination;
            next_rr_destination = (next_rr_destination + 1) % dest2Socket.size();
        }


        sendToSck(dest2Socket[task->chid], task);
        delete task;
        return this->GO_ON;
    }

    void eosnotify(ssize_t id) {
        // receive it from an internal gateway
        if (internalGateways > 0 && id >= (ssize_t)(this->get_num_inchannels() - internalGateways) && ++Ineos == internalGateways){
            message_t E_O_S(0,0);
            for (const auto & sck : type2sck[ConnectionType::INTERNAL]) sendToSck(sck, &E_O_S);
        } else 
            ++neos;

        if (neos > 0 && (neos + Ineos) >= (int)this->get_num_inchannels()){
            // send to all the external
            message_t E_O_S(0,0);
            for(const auto& sck : type2sck[ConnectionType::EXTERNAL]) sendToSck(sck, &E_O_S);
        }
    }

};



/*
    ONDEMAND specification
*/

class ff_dsenderOD: public ff_dsender { 
private:
    int last_rr_socket = 0; //next destiation to send for round robin policy
    std::map<int, unsigned int> sockCounters;
    const int queueDim;
    fd_set set, tmpset;
    int fdmax = -1;

    
public:
    ff_dsenderOD(ff_endpoint dest_endpoint, int queueDim = 1, int coreid=-1)
		: ff_dsender(dest_endpoint, coreid), queueDim(queueDim) {}

    ff_dsenderOD(std::vector<ff_endpoint> dest_endpoints_, int queueDim = 1, int coreid=-1)
		: ff_dsender(dest_endpoints_, coreid), queueDim(queueDim) {}

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
		
        FD_ZERO(&set);
        FD_ZERO(&tmpset);

		sockets.resize(this->dest_endpoints.size());
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
            for (size_t i = 0; i < this->sockets.size(); ++i){
                int r; ack_t a;
                if ((r = recvnnb(sockets[i], reinterpret_cast<char*>(&a), sizeof(ack_t))) != sizeof(ack_t)){
                    if (errno == EWOULDBLOCK){
                        assert(r == -1);
                        continue;
                    }
                    perror("recvnnb ack");
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

	int waitAckFromAny() {
		tmpset = set;
		if (select(fdmax + 1, &tmpset, NULL, NULL, NULL) == -1){
			perror("select");
			return -1;
		} 
		// try to receive from all connections in a non blocking way
		for (size_t i = 0; i < this->sockets.size(); ++i){
			int r; ack_t a;
			int sck = sockets[i];
			if ((r = recvnnb(sck, reinterpret_cast<char*>(&a), sizeof(ack_t))) != sizeof(ack_t)){
				if (errno == EWOULDBLOCK){
					assert(r == -1);
					continue;
				}
				perror("recvnnb ack");
				return -1;
			} else {
				sockCounters[sck]++;
				last_rr_socket = i;
				return sck;
			}
		} 
		assert(1==0);
		return -1;
	}

    int getNextReady(){
        for(size_t i = 0; i < this->sockets.size(); i++){
            int actualSocket = (last_rr_socket + 1 + i) % this->sockets.size();
            int sck = sockets[actualSocket];
            if (sockCounters[sck] > 0) {
                last_rr_socket = actualSocket;
                return sck;
            }
        }
		return waitAckFromAny();		
    }


    void svc_end() {
		long totalack = sockets.size()*queueDim;
		long currack  = 0;
        for(const auto& pair : sockCounters)
            currack += pair.second;
		while(currack<totalack) {
			waitAckFromAny();
			currack++;
		}

		// close the socket not matter if local or remote
        for(size_t i=0; i < this->sockets.size(); i++) {
	            close(sockets[i]);
		}
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


};

#endif
