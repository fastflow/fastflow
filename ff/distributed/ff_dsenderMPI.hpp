#ifndef FF_DSENDER_MPI_H
#define FF_DSENDER_MPI_H

#include <iostream>
#include <map>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
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
protected:
    size_t neos=0;
    int next_rr_destination = 0; //next destiation to send for round robin policy
    std::map<int, int> dest2Rank;
    std::vector<ff_endpoint> destRanks;
	int coreid;

    int receiveReachableDestinations(int rank){
        int sz;
        int cmd = DFF_REQUEST_ROUTING_TABLE;
        
        MPI_Status status;
        MPI_Send(&cmd, 1, MPI_INT, rank, DFF_ROUTING_TABLE_REQUEST_TAG, MPI_COMM_WORLD);
        MPI_Probe(rank, DFF_ROUTING_TABLE_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status,MPI_BYTE, &sz);
        char* buff = new char [sz];
        std::cout << "Received routing table (" << sz  << " bytes)" << std::endl;
        MPI_Recv(buff, sz, MPI_BYTE, rank, DFF_ROUTING_TABLE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        dataBuffer dbuff(buff, sz, true);
        std::istream iss(&dbuff);
		cereal::PortableBinaryInputArchive iarchive(iss);
        std::vector<int> destinationsList;

        iarchive >> destinationsList;

        for (int d : destinationsList) dest2Rank[d] = rank;

        return 0;
    }

    
public:
    ff_dsenderMPI(ff_endpoint destRank, int coreid=-1)
		: coreid(coreid) {
        this->destRanks.push_back(std::move(destRank));
    }

    ff_dsenderMPI( std::vector<ff_endpoint> destRanks_, int coreid=-1)
		: destRanks(std::move(destRanks_)),coreid(coreid) {}

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);

        for(ff_endpoint& ep: this->destRanks)
           receiveReachableDestinations(ep.getRank()); 

        return 0;
    }

  
    message_t *svc(message_t* task) {
        /* here i should send the task via socket */
        if (task->chid == -1){ // roundrobin over the destinations
            task->chid = next_rr_destination;
            next_rr_destination = (next_rr_destination + 1) % dest2Rank.size();
        }

        size_t sz = task->data.getLen();
        int rank =dest2Rank[task->chid];
        
        long header[3] = {(long)sz, task->sender, task->chid};

        MPI_Send(header, 3, MPI_LONG, rank, DFF_HEADER_TAG, MPI_COMM_WORLD);
    
        MPI_Send(task->data.getPtr(), sz, MPI_BYTE, rank, DFF_TASK_TAG, MPI_COMM_WORLD);


        delete task;
        return this->GO_ON;
    }

     void eosnotify(ssize_t) {
	    if (++neos >= this->get_num_inchannels()){
            long header[3] = {0,0,0};
            
            for(auto& ep : destRanks)
                MPI_Send(header, 3, MPI_LONG, ep.getRank(), DFF_HEADER_TAG, MPI_COMM_WORLD);

        }
    }
};


/* versione Ondemand */

class ff_dsenderMPIOD: public ff_dsenderMPI { 
private:
    int last_rr_rank = 0; //next destiation to send for round robin policy
    std::map<int, unsigned int> rankCounters;
    int queueDim;
    
public:
    ff_dsenderMPIOD(ff_endpoint destRank, int queueDim_ = 1, int coreid=-1)
		: ff_dsenderMPI(destRank, coreid), queueDim(queueDim_) {}

    ff_dsenderMPIOD( std::vector<ff_endpoint> destRanks_, int queueDim_ = 1, int coreid=-1)
		: ff_dsenderMPI(destRanks_, coreid), queueDim(queueDim_){}

    int svc_init() {
        std::cout << "instantiating the ondemand mpi sender!\n";

		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);

        for(ff_endpoint& ep: this->destRanks){
           receiveReachableDestinations(ep.getRank());
           rankCounters[ep.getRank()] = queueDim;
        } 

        return 0;
    }

    void svc_end(){
        long totalack = destRanks.size()*queueDim;
		long currack  = 0;
        
        for(const auto& pair : rankCounters) currack += pair.second;
		
        while(currack<totalack) {
			waitAckFromAny();
			currack++;
		}
    }

    inline int waitAckFromAny(){
        ack_t tmpAck;
        MPI_Status status;
        if (MPI_Recv(&tmpAck, sizeof(ack_t), MPI_BYTE, MPI_ANY_SOURCE, DFF_ACK_TAG, MPI_COMM_WORLD, &status) != MPI_SUCCESS)
            return -1;
        rankCounters[status.MPI_SOURCE]++;
        return status.MPI_SOURCE;
    }

    int waitAckFrom(int rank){
        ack_t tmpAck;
        MPI_Status status;
        while(true){
            if (MPI_Recv(&tmpAck, sizeof(ack_t), MPI_BYTE, MPI_ANY_SOURCE, DFF_ACK_TAG, MPI_COMM_WORLD, &status) != MPI_SUCCESS)
                return -1;
            rankCounters[status.MPI_SOURCE]++;
            if (rank == status.MPI_SOURCE) return 0;
        }
    }

    int getNextReady(){
        for(size_t i = 0; i < this->destRanks.size(); i++){
            int rankIndex = (last_rr_rank + 1 + i) % this->destRanks.size();
            int rank = destRanks[rankIndex].getRank();
            if (rankCounters[rank] > 0) {
                last_rr_rank = rankIndex;
                return rank;
            }
        }
		return waitAckFromAny();		
    }

  
    message_t *svc(message_t* task) {
        int rank;
        if (task->chid != -1){
            rank = dest2Rank[task->chid];
            if (rankCounters[rank] == 0 && waitAckFrom(rank) == -1){
                error("Error waiting ACK\n");
                delete task; return this->GO_ON;
            }
        } else {
            rank = getNextReady();
        }

        size_t sz = task->data.getLen();
        
        long header[3] = {(long)sz, task->sender, task->chid};

        MPI_Send(header, 3, MPI_LONG, rank, DFF_HEADER_TAG, MPI_COMM_WORLD);
    
        MPI_Send(task->data.getPtr(), sz, MPI_BYTE, rank, DFF_TASK_TAG, MPI_COMM_WORLD);

        rankCounters[rank]--;

        delete task;
        return this->GO_ON;
    }
};

#endif
