#ifndef FF_DRECEIVER_MPI_H
#define FF_DRECEIVER_MPI_H


#include <iostream>
#include <sstream>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <mpi.h>
#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>

using namespace ff;

class ff_dreceiverMPI: public ff_monode_t<message_t> { 
private:

    int sendRoutingTable(int rank){
        dataBuffer buff; std::ostream oss(&buff);
		cereal::PortableBinaryOutputArchive oarchive(oss);
        std::vector<int> reachableDestinations;

        for(auto const& p : this->routingTable) reachableDestinations.push_back(p.first);

		oarchive << reachableDestinations;

        /*size_t sz = htobe64(buff.getLen());
        struct iovec iov[1];
        iov[0].iov_base = &sz;
        iov[0].iov_len = sizeof(sz);

        if (writevn(sck, iov, 1) < 0 || writen(sck, buff.getPtr(), buff.getLen()) < 0){
            error("Error writing on socket the routing Table\n");
            return -1;
        }*/

        if (MPI_Send(buff.getPtr(), buff.getLen(), MPI_BYTE, rank, DFF_ROUTING_TABLE_TAG, MPI_COMM_WORLD) != MPI_SUCCESS)
            return -1;

        return 0;
    }

public:
    ff_dreceiverMPI(size_t input_channels, std::map<int, int> routingTable = {std::make_pair(0,0)}, int coreid=-1)
		: input_channels(input_channels), routingTable(routingTable), coreid(coreid) {}

    int svc_init() {
  		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);

        /*#ifdef LOCAL
            if ((listen_sck=socket(AF_LOCAL, SOCK_STREAM, 0)) < 0){
                error("Error creating the socket\n");
                return -1;
            }
            
            struct sockaddr_un serv_addr;
            memset(&serv_addr, '0', sizeof(serv_addr));
            serv_addr.sun_family = AF_LOCAL;
            strncpy(serv_addr.sun_path, acceptAddr.address.c_str(), acceptAddr.address.size()+1);
        #endif

        #ifdef REMOTE
            if ((listen_sck=socket(AF_INET, SOCK_STREAM, 0)) < 0){
                error("Error creating the socket\n");
                return -1;
            }

            int enable = 1;
            // enable the reuse of the address
            if (setsockopt(listen_sck, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
                error("setsockopt(SO_REUSEADDR) failed\n");

            struct sockaddr_in serv_addr;
            serv_addr.sin_family = AF_INET; 
            serv_addr.sin_addr.s_addr = INADDR_ANY; // still listening from any interface
            serv_addr.sin_port = htons( acceptAddr.port );

        #endif

        if (bind(listen_sck, (struct sockaddr*)&serv_addr,sizeof(serv_addr)) < 0){
            error("Error binding\n");
            return -1;
        }

        if (listen(listen_sck, MAXBACKLOG) < 0){
            error("Error listening\n");
            return -1;
        }

        /*for (const auto& e : routingTable)
            std::cout << "Entry: " << e.first << " -> " << e.second << std::endl;
        */

        return 0;
    }

    /* 
        Here i should not care of input type nor input data since they come from a socket listener.
        Everything will be handled inside a while true in the body of this node where data is pulled from network
    */
    message_t *svc(message_t* task) {
        MPI_Status status;
        char headerBuff[sizeof(size_t)+2*sizeof(int)];
        while(neos < input_channels){
            MPI_Recv(headerBuff, sizeof(size_t)+2*sizeof(int), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == DFF_ROUTING_TABLE_TAG){
                sendRoutingTable(status.MPI_SOURCE);
                continue;
            }

            size_t sz = *reinterpret_cast<size_t*>(headerBuff);

            if (sz == 0){
                neos++;
                continue;
            }

            char* buff = new char [sz];
			assert(buff);
            
            MPI_Recv(buff,sz,MPI_BYTE, status.MPI_SOURCE, DFF_TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            message_t* out = new message_t(buff, sz, true);
			assert(out);
			out->sender = *reinterpret_cast<int*>(headerBuff+sizeof(size_t));
			out->chid   = *reinterpret_cast<int*>(headerBuff+sizeof(size_t)+sizeof(int));

            //std::cout << "received something from " << sender << " directed to " << chid << std::endl;

            ff_send_out_to(out, this->routingTable[out->chid]); // assume the routing table is consistent WARNING!!!
        }
        
        
        
        
        /* here i should receive the task via socket */
        
        /*fd_set set, tmpset;
        // intialize both sets (master, temp)
        FD_ZERO(&set);
        FD_ZERO(&tmpset);

        // add the listen socket to the master set
        FD_SET(this->listen_sck, &set);

        // hold the greater descriptor
        int fdmax = this->listen_sck; 

        while(neos < input_channels){
            // copy the master set to the temporary
            tmpset = set;

            switch(select(fdmax+1, &tmpset, NULL, NULL, NULL)){
                case -1: error("Error on selecting socket\n"); return EOS;
                case  0: continue;
            }

            // iterate over the file descriptor to see which one is active
            for(int i=0; i <= fdmax; i++) 
	            if (FD_ISSET(i, &tmpset)){
                    if (i == this->listen_sck) {
                        int connfd = accept(this->listen_sck, (struct sockaddr*)NULL ,NULL);
                        if (connfd == -1){
                            error("Error accepting client\n");
                        } else {
                            FD_SET(connfd, &set);
                            if(connfd > fdmax) fdmax = connfd;

                            this->sendRoutingTable(connfd); // here i should check the result of the call! and handle possible errors!
                        }
                        continue;
                    }
                    
                    // it is not a new connection, call receive and handle possible errors
                    if (this->handleRequest(i) < 0){
                        close(i);
                        FD_CLR(i, &set);

                        // update the maximum file descriptor
                        if (i == fdmax)
                            for(int i=(fdmax-1);i>=0;--i)
                                if (FD_ISSET(i, &set)){
                                    fdmax = i;
                                    break;
                                }
                                    
                    }
                }

        }

        /* In theory i should never return because of the while true. In our first example this is necessary */
        return this->EOS;
    }

private:
    size_t neos = 0;
    size_t input_channels;
    std::map<int, int> routingTable;
	int coreid;
};

#endif
