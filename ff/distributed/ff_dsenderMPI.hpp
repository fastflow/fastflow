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

#include <ff/distributed/ff_batchbuffer.hpp>


using namespace ff;

class ff_dsenderMPI: public ff_minode_t<message_t> { 
protected:
    size_t neos=0;
    int last_rr_rank = 0; //next destiation to send for round robin policy
    std::map<int, int> dest2Rank;
    std::map<int, unsigned int> rankCounters;
    std::map<int, ff_batchBuffer> batchBuffers;
    std::vector<ff_endpoint> destRanks;
    std::string gName;
    int batchSize;
    int messageOTF;
	int coreid;

    static int receiveReachableDestinations(int rank, std::map<int, int>& m){
        int sz;
        //int cmd = DFF_REQUEST_ROUTING_TABLE;
        
        MPI_Status status;
        //MPI_Send(&cmd, 1, MPI_INT, rank, DFF_ROUTING_TABLE_REQUEST_TAG, MPI_COMM_WORLD);
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

        for (const int& d : destinationsList) m[d] = rank;

        return 0;
    }

    virtual int handshakeHandler(const int rank, bool){
        MPI_Send(gName.c_str(), gName.size(), MPI_BYTE, rank, DFF_GROUP_NAME_TAG, MPI_COMM_WORLD);

        return receiveReachableDestinations(rank, dest2Rank);
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

    int waitAckFromAny(){
        ack_t tmpAck;
        MPI_Status status;
        if (MPI_Recv(&tmpAck, sizeof(ack_t), MPI_BYTE, MPI_ANY_SOURCE, DFF_ACK_TAG, MPI_COMM_WORLD, &status) != MPI_SUCCESS)
            return -1;
        rankCounters[status.MPI_SOURCE]++;
        return status.MPI_SOURCE;
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

public:
    ff_dsenderMPI(ff_endpoint destRank, std::string gName = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int coreid=-1)
		: gName(gName), batchSize(batchSize), messageOTF(messageOTF), coreid(coreid) {
        this->destRanks.push_back(std::move(destRank));
    }

    ff_dsenderMPI( std::vector<ff_endpoint> destRanks_, std::string gName = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int coreid=-1)
		: destRanks(std::move(destRanks_)), gName(gName), batchSize(batchSize), messageOTF(messageOTF), coreid(coreid) {}

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);

        for(ff_endpoint& ep: this->destRanks){
           handshakeHandler(ep.getRank(), false);
           rankCounters[ep.getRank()] = messageOTF;

        }

        return 0;
    }

    void svc_end(){
        long totalack = destRanks.size() * messageOTF;
		long currack  = 0;
        
        for(const auto& pair : rankCounters) currack += pair.second;
		
        while(currack<totalack) {
			waitAckFromAny();
			currack++;
		}
    }
  
    message_t *svc(message_t* task) {
        int rank;
        if (task->chid != -1){
            rank = dest2Rank[task->chid];
            if (rankCounters[rank] == 0 && waitAckFrom(rank) == -1){
                error("Error waiting ACK\n");
                delete task; return this->GO_ON;
            }
        } else 
            rank = getNextReady();

        size_t sz = task->data.getLen();
        
        long header[3] = {(long)sz, task->sender, task->chid};

        MPI_Send(header, 3, MPI_LONG, rank, DFF_HEADER_TAG, MPI_COMM_WORLD);
    
        MPI_Send(task->data.getPtr(), sz, MPI_BYTE, rank, DFF_TASK_TAG, MPI_COMM_WORLD);

        rankCounters[rank]--;

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

#endif
