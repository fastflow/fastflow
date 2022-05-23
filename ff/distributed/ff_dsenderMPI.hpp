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
    class batchBuffer {
    protected:    
        int rank;
        bool blocked = false;
        size_t size_, actualSize = 0;
        std::vector<char> buffer;
        std::vector<long> headers;
        MPI_Request headersR, datasR;
    public:
        batchBuffer(size_t size_, int rank) : rank(rank), size_(size_){
            headers.reserve(size_*3+1);
        }
        virtual void waitCompletion(){
            if (blocked){
                MPI_Wait(&headersR, MPI_STATUS_IGNORE);
                MPI_Wait(&datasR, MPI_STATUS_IGNORE);
                headers.clear();
                buffer.clear();
                blocked = false;
            }
        }

        virtual size_t size() {return actualSize;}
        virtual int push(message_t* m){
            waitCompletion();
            int idx = 3*actualSize++;
            headers[idx+1] = m->sender;
            headers[idx+2] = m->chid;
            headers[idx+3] = m->data.getLen();

            buffer.insert(buffer.end(), m->data.getPtr(), m->data.getPtr() + m->data.getLen());

            delete m;
            if (actualSize == size_) {
                this->flush();
                return 1;
            }
            return 0;
        }

        virtual void flush(){
            headers[0] = actualSize;
            MPI_Isend(headers.data(), actualSize*3+1, MPI_LONG, rank, DFF_HEADER_TAG, MPI_COMM_WORLD, &headersR);
            MPI_Isend(buffer.data(), buffer.size(), MPI_BYTE, rank, DFF_TASK_TAG, MPI_COMM_WORLD, &datasR);
            blocked = true;
            actualSize = 0;
        }

        virtual void pushEOS(){
            int idx = 3*actualSize++;
            headers[idx+1] = 0; headers[idx+2] = 0; headers[idx+3] = 0;

            this->flush();
        }
    };

    class directBatchBuffer : public batchBuffer {
            message_t* currData = NULL;
            long currHeader[4] = {1, 0, 0, 0}; 
        public:
            directBatchBuffer(int rank) : batchBuffer(0, rank){}
            void pushEOS(){
                waitCompletion();
                currHeader[1] = 0; currHeader[2] = 0; currHeader[3] = 0;
                MPI_Send(currHeader, 4, MPI_LONG, this->rank, DFF_HEADER_TAG, MPI_COMM_WORLD);
            }
            void flush() {}
            void waitCompletion(){
                if (blocked){
                    MPI_Wait(&headersR, MPI_STATUS_IGNORE);
                    MPI_Wait(&datasR, MPI_STATUS_IGNORE);
                    if (currData) delete currData;
                    blocked = false;
                }
            }
            int push(message_t* m){
                waitCompletion();
                currHeader[1] = m->sender; currHeader[2] = m->chid; currHeader[3] = m->data.getLen();
                MPI_Isend(currHeader, 4, MPI_LONG, this->rank, DFF_HEADER_TAG, MPI_COMM_WORLD, &this->headersR);
                MPI_Isend(m->data.getPtr(), m->data.getLen(), MPI_BYTE, rank, DFF_TASK_TAG, MPI_COMM_WORLD, &datasR);
                currData = m;
                blocked = true;
                return 1;
            }
    };
    size_t neos=0;
    int last_rr_rank = 0; //next destiation to send for round robin policy
    std::map<int, int> dest2Rank;
    std::map<int, unsigned int> rankCounters;
    std::map<int, std::pair<int, std::vector<batchBuffer*>>> buffers;
    std::vector<int> ranks;
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

    int sendGroupName(const int rank){    
        MPI_Send(gName.c_str(), gName.size(), MPI_BYTE, rank, DFF_GROUP_NAME_TAG, MPI_COMM_WORLD);
        return 0;
    }

    virtual int handshakeHandler(const int rank, bool){
        sendGroupName(rank);
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
        for(size_t i = 0; i < this->ranks.size(); i++){
            int rankIndex = (last_rr_rank + 1 + i) % this->ranks.size();
            int rank = ranks[rankIndex];
            if (rankCounters[rank] > 0) {
                last_rr_rank = rankIndex;
                return rank;
            }
        }
		return waitAckFromAny();		
    }

    int sendToRank(const int rank, const message_t* task){
        size_t sz = task->data.getLen();
        
        long header[3] = {(long)sz, task->sender, task->chid};

        MPI_Send(header, 3, MPI_LONG, rank, DFF_HEADER_TAG, MPI_COMM_WORLD);
    
        MPI_Send(task->data.getPtr(), sz, MPI_BYTE, rank, DFF_TASK_TAG, MPI_COMM_WORLD);

        return 0;

    }

    int getMostFilledBufferRank(){
        int rankMax = -1;
        size_t sizeMax = 0;
        for(int rank : ranks){
            auto& batchBB = buffers[rank];
            size_t sz = batchBB.second[batchBB.first]->size();
            if (sz > sizeMax) {
                rankMax = rank;
                sizeMax = sz;
            }
        }
        if (rankMax >= 0) return rankMax;

        last_rr_rank = (last_rr_rank + 1) % this->ranks.size();
        return this->ranks[last_rr_rank]; 
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
           //rankCounters[ep.getRank()] = messageOTF;
           ranks.push_back(ep.getRank());
           std::vector<batchBuffer*> appo;
           for(int i = 0; i < messageOTF; i++) appo.push_back(batchSize == 1 ? new directBatchBuffer(ep.getRank()) : new batchBuffer(batchSize, ep.getRank()));
           buffers.emplace(std::make_pair(ep.getRank(), std::make_pair(0, std::move(appo))));
        }

         this->destRanks.clear();

        return 0;
    }

    void svc_end(){
        for(auto& [rank, bb] : buffers)
            for(auto& b : bb.second) b->waitCompletion();
    }
  
    message_t *svc(message_t* task) {
        int rank;
        if (task->chid != -1)
            rank = dest2Rank[task->chid]; 
        else 
            rank = getMostFilledBufferRank();
        
        auto& buffs = buffers[rank];
        assert(buffs.second.size() > 0);
        if (buffs.second[buffs.first]->push(task)) // the push triggered a flush, so we must go ion the next buffer
            buffs.first = (buffs.first + 1) % buffs.second.size(); // increment the used buffer of 1
    
        return this->GO_ON;
    }

     void eosnotify(ssize_t) {
	    if (++neos >= this->get_num_inchannels())
            for(auto& rank : ranks){
                auto& buffs = buffers[rank];
                buffs.second[buffs.first]->pushEOS();
            }       
    }
};


class ff_dsenderHMPI : public ff_dsenderMPI {

    std::map<int, int> internalDest2Rank;
    std::vector<int> internalRanks;
    int last_rr_rank_Internal = -1;
    std::set<std::string> internalGroupNames;
    int internalMessageOTF;
    bool squareBoxEOS = false;

    int getNextReadyInternal(){
        for(size_t i = 0; i < this->internalRanks.size(); i++){
            int actualRankIndex = (last_rr_rank_Internal + 1 + i) % this->internalRanks.size();
            int sck = internalRanks[actualRankIndex];
            if (rankCounters[sck] > 0) {
                last_rr_rank_Internal = actualRankIndex;
                return sck;
            }
        }

        int rank;
        decltype(internalRanks)::iterator it;

        do 
            rank = waitAckFromAny();   // FIX: error management!
        while ((it = std::find(internalRanks.begin(), internalRanks.end(), rank)) != internalRanks.end());
        
        last_rr_rank_Internal = it - internalRanks.begin();
        return rank;
    }

    int getMostFilledInternalBufferRank(){
        int rankMax = -1;
        size_t sizeMax = 0;
        for(int rank : internalRanks){
            auto& batchBB = buffers[rank];
            size_t sz = batchBB.second[batchBB.first]->size();
            if (sz > sizeMax) {
                rankMax = rank;
                sizeMax = sz;
            }
        }
        if (rankMax >= 0) return rankMax;

        last_rr_rank_Internal = (last_rr_rank_Internal + 1) % this->internalRanks.size();
        return internalRanks[last_rr_rank_Internal];
    }

public:

    ff_dsenderHMPI(ff_endpoint e, std::string gName  = "", std::set<std::string> internalGroups = {}, int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int internalMessageOTF = DEFAULT_INTERNALMSG_OTF, int coreid=-1) : ff_dsenderMPI(e, gName, batchSize, messageOTF, coreid), internalGroupNames(internalGroups), internalMessageOTF(internalMessageOTF) {} 
    ff_dsenderHMPI(std::vector<ff_endpoint> dest_endpoints_, std::string gName  = "", std::set<std::string> internalGroups = {}, int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int internalMessageOTF = DEFAULT_INTERNALMSG_OTF, int coreid=-1) : ff_dsenderMPI(dest_endpoints_, gName, batchSize, messageOTF, coreid), internalGroupNames(internalGroups), internalMessageOTF(internalMessageOTF) {}
    
    int handshakeHandler(const int rank, bool isInternal){
        sendGroupName(rank);

        return receiveReachableDestinations(rank, isInternal ? internalDest2Rank : dest2Rank);
    }

    int svc_init() {

        if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
		
        for(auto& endpoint : this->destRanks){
            int rank = endpoint.getRank();
            bool isInternal = internalGroupNames.contains(endpoint.groupName);
            if (isInternal) 
                internalRanks.push_back(rank);
            else
                ranks.push_back(rank);

            std::vector<batchBuffer*> appo;
            for(int i = 0; i < (isInternal ? internalMessageOTF : messageOTF); i++) appo.push_back(batchSize == 1 ? new directBatchBuffer(rank) : new batchBuffer(batchSize, rank));
            buffers.emplace(std::make_pair(rank, std::make_pair(0, std::move(appo))));
            
            if (handshakeHandler(rank, isInternal) < 0) return -1;

        }

        this->destRanks.clear();

        return 0;
    }

    message_t *svc(message_t* task) {
        if (this->get_channel_id() == (ssize_t)(this->get_num_inchannels() - 1)){
            int rank;
        
            // pick destination from the list of internal connections!
            if (task->chid != -1){ // roundrobin over the destinations
                rank = internalDest2Rank[task->chid];
            } else
                rank = getMostFilledInternalBufferRank();


            auto& buffs = buffers[rank];
            if (buffs.second[buffs.first]->push(task)) // the push triggered a flush, so we must go ion the next buffer
                buffs.first = (buffs.first + 1) % buffs.second.size(); // increment the used buffer of 1

            return this->GO_ON;
        }
        
        return ff_dsenderMPI::svc(task);
    }

     void eosnotify(ssize_t id) {
         if (id == (ssize_t)(this->get_num_inchannels() - 1)){
            // send the EOS to all the internal connections
            if (squareBoxEOS) return;
            squareBoxEOS = true;
            for(const auto&rank : internalRanks){
                auto& buffs = buffers[rank];
                buffs.second[buffs.first]->pushEOS();
            }
		 }
		 if (++neos >= this->get_num_inchannels()) {
			 // all input EOS received, now sending the EOS to all
			 // others connections
			 for(const auto& rank : ranks){
                auto& buffs = buffers[rank];
                buffs.second[buffs.first]->pushEOS();
             }
		 }
	 }

	void svc_end(){
        for(auto& [rank, bb] : buffers)
            for(auto& b : bb.second) b->waitCompletion();
    }
	
};

#endif
