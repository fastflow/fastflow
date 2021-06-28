#ifndef FF_DSENDER_MPI_H
#define FF_DSENDER_MPI_H

#include <iostream>
#include <map>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <cmath>
#include <thread>
#include <mpi.h>

#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>


using namespace ff;

class ff_dsenderMPI: public ff_minode_t<message_t> { 
private:
    size_t neos=0;
    int next_rr_destination = 0; //next destiation to send for round robin policy
    std::map<int, int> dest2Rank;
    std::vector<int> destRanks;
	int coreid;

    int receiveReachableDestinations(int rank){
        int sz;
        int cmd = DFF_REQUEST_ROUTING_TABLE;
        char* buff = new char [1000];

        MPI_Status status;
        MPI_Sendrecv(&cmd, 1, MPI_INT, rank, DFF_ROUTING_TABLE_TAG, buff, 1000, MPI_BYTE, rank, DFF_ROUTING_TABLE_TAG, MPI_COMM_WORLD, &status);

        MPI_Get_count(&status,MPI_BYTE, &sz);

        std::cout << "Received routing table (" << sz  << " bytes)" << std::endl;
     
        dataBuffer dbuff(buff, sz, true);
        std::istream iss(&dbuff);
		cereal::PortableBinaryInputArchive iarchive(iss);
        std::vector<int> destinationsList;

        iarchive >> destinationsList;

        for (int d : destinationsList) dest2Rank[d] = rank;

        return 0;
    }
	

    int sendToSck(int rank, message_t* task){
        //std::cout << "received something from " << task->sender << " directed to " << task->chid << std::endl;
        size_t sz = task->data.getLen();
        
        char headerBuff[sizeof(size_t)+2*sizeof(int)];
        memcpy(headerBuff, &sz, sizeof(size_t));
        memcpy(headerBuff+sizeof(size_t), &task->sender, sizeof(int));
        memcpy(headerBuff+sizeof(int)+sizeof(size_t), &task->chid, sizeof(int));
        
        if (MPI_Send(headerBuff, sizeof(size_t)+2*sizeof(int), MPI_BYTE, rank, DFF_HEADER_TAG, MPI_COMM_WORLD) != MPI_SUCCESS)
            return -1;
        if (sz > 0)
            if (MPI_Send(task->data.getPtr(), sz, MPI_BYTE, rank, DFF_TASK_TAG, MPI_COMM_WORLD) != MPI_SUCCESS)
                return -1;

        return 0;
    }

    
public:
    ff_dsenderMPI(int destRank, int coreid=-1)
		: coreid(coreid) {
        this->destRanks.push_back(std::move(destRank));
    }

    ff_dsenderMPI( std::vector<int> destRanks_, int coreid=-1)
		: destRanks(std::move(destRanks_)),coreid(coreid) {}

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);

        for(const int& rank: this->destRanks)
           receiveReachableDestinations(rank); 

        return 0;
    }

  
    message_t *svc(message_t* task) {
        /* here i should send the task via socket */
        if (task->chid == -1){ // roundrobin over the destinations
            task->chid = next_rr_destination;
            next_rr_destination = (next_rr_destination + 1) % dest2Rank.size();
        }

        sendToSck(dest2Rank[task->chid], task);

        delete task;
        return this->GO_ON;
    }

     void eosnotify(ssize_t) {
	    if (++neos >= this->get_num_inchannels()){
            message_t * E_O_S = new message_t;
            E_O_S->chid = 0;
            E_O_S->sender = 0;
            for(const auto& rank : destRanks)
                sendToSck(rank, E_O_S);

            delete E_O_S;
        }
    }


};

#endif
