/* ***************************************************************************
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */
/* Authors: 
 *   Nicolo' Tonci
 *   Massimo Torquati
 */

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
    size_t neos=0;
    int next_rr_destination = 0; //next destiation to send for round robin policy
    std::vector<ff_endpoint> dest_endpoints;
    std::map<int, int> dest2Socket;
    //std::unordered_map<ConnectionType, std::vector<int>> type2sck;
    std::vector<int> sockets;
	//int internalGateways;
    std::string gName;
    int coreid;

    int receiveReachableDestinations(int sck, std::map<int,int>& m){
       
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

        for (const int& d : destinationsList) m[d] = sck;

        /*for (const auto& p : m)
            std::cout << p.first << " - "  << p.second << std::endl;
        */
       ff::cout << "Receiving routing table (" << sz << " bytes)" << ff::endl;
        return 0;
    }

    int sendGroupName(const int sck){    
        size_t sz = htobe64(gName.size());
        struct iovec iov[2];
        iov[0].iov_base = &sz;
        iov[0].iov_len = sizeof(sz);
        iov[1].iov_base = (char*)(gName.c_str());
        iov[1].iov_len = gName.size();

        if (writevn(sck, iov, 2) < 0){
            error("Error writing on socket\n");
            return -1;
        }

        return 0;
    }

    virtual int handshakeHandler(const int sck, bool){
        if (sendGroupName(sck) < 0) return -1;

        return receiveReachableDestinations(sck, dest2Socket);
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


        // receive the reachable destination from this sockets

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
    ff_dsender(ff_endpoint dest_endpoint, std::string gName = "", int coreid=-1): gName(gName), coreid(coreid) {
        this->dest_endpoints.push_back(std::move(dest_endpoint));
    }

    ff_dsender( std::vector<ff_endpoint> dest_endpoints_, std::string gName = "", int coreid=-1) : dest_endpoints(std::move(dest_endpoints_)), gName(gName), coreid(coreid) {}

    

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
		
        sockets.resize(this->dest_endpoints.size());
        for(size_t i=0; i < this->dest_endpoints.size(); i++){
            if ((sockets[i] = tryConnect(this->dest_endpoints[i])) <= 0 ) return -1;
            if (handshakeHandler(sockets[i], false) < 0) return -1;
        }

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
        if (++neos >= this->get_num_inchannels()){
            message_t E_O_S(0,0);
            for(const auto& sck : sockets) sendToSck(sck, &E_O_S);
        }
    }

};


class ff_dsenderH : public ff_dsender {

    std::map<int, int> internalDest2Socket;
    std::map<int, int>::const_iterator rr_iterator;
    std::vector<int> internalSockets;
    std::set<std::string> internalGroupNames;

public:

    ff_dsenderH(ff_endpoint e, std::string gName  = "", std::set<std::string> internalGroups = {}, int coreid=-1) : ff_dsender(e, gName, coreid), internalGroupNames(internalGroups) {} 
    ff_dsenderH(std::vector<ff_endpoint> dest_endpoints_, std::string gName  = "", std::set<std::string> internalGroups = {}, int coreid=-1) : ff_dsender(dest_endpoints_, gName, coreid), internalGroupNames(internalGroups) {}
    
    int handshakeHandler(const int sck, bool isInternal){
        if (sendGroupName(sck) < 0) return -1;

        return receiveReachableDestinations(sck, isInternal ? internalDest2Socket : dest2Socket);
    }

    int svc_init() {

        sockets.resize(this->dest_endpoints.size());
        for(const auto& endpoint : this->dest_endpoints){
            int sck = tryConnect(endpoint);
            if (sck <= 0) {
                error("Error on connecting!\n");
                return -1;
            }
            bool isInternal = internalGroupNames.contains(endpoint.groupName);
            if (isInternal) internalSockets.push_back(sck);
            else sockets.push_back(sck);
            handshakeHandler(sck, isInternal);
        }

        rr_iterator = internalDest2Socket.cbegin();
        return 0;
    }

    message_t *svc(message_t* task) {
        if (this->get_channel_id() == (ssize_t)(this->get_num_inchannels() - 1)){
            // pick destination from the list of internal connections!
            if (task->chid == -1){ // roundrobin over the destinations
                task->chid = rr_iterator->first;
                if (++rr_iterator == internalDest2Socket.cend()) rr_iterator = internalDest2Socket.cbegin();
            }

            sendToSck(internalDest2Socket[task->chid], task); 
            delete task;
            return this->GO_ON;
        }

        return ff_dsender::svc(task);
    }

     void eosnotify(ssize_t id) {
         if (id == (ssize_t)(this->get_num_inchannels() - 1)){
            // send the EOS to all the internal connections
            message_t E_O_S(0,0);
            for(const auto& sck : internalSockets) sendToSck(sck, &E_O_S);            
         }
		 if (++neos >= this->get_num_inchannels()){
			 message_t E_O_S(0,0);
			 for(const auto& sck : sockets) sendToSck(sck, &E_O_S);
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
    ff_dsenderOD(ff_endpoint dest_endpoint, int queueDim = 1, std::string gName = "", int coreid=-1)
		: ff_dsender(dest_endpoint, gName, coreid), queueDim(queueDim) {}

    ff_dsenderOD(std::vector<ff_endpoint> dest_endpoints_, int queueDim = 1, std::string gName = "", int coreid=-1)
		: ff_dsender(dest_endpoints_, gName, coreid), queueDim(queueDim) {}

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
		
        FD_ZERO(&set);
        FD_ZERO(&tmpset);

		sockets.resize(this->dest_endpoints.size());
        for(size_t i=0; i < this->dest_endpoints.size(); i++){
            if ((sockets[i] = tryConnect(this->dest_endpoints[i])) <= 0 ) return -1;
			if (handshakeHandler(sockets[i], false) < 0) return -1;
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
