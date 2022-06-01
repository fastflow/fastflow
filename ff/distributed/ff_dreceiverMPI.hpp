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
protected:

    static int sendRoutingTable(const int rank, const std::vector<int>& dest){
        dataBuffer buff; std::ostream oss(&buff);
		cereal::PortableBinaryOutputArchive oarchive(oss);
		oarchive << dest;

        if (MPI_Send(buff.getPtr(), buff.getLen(), MPI_BYTE, rank, DFF_ROUTING_TABLE_TAG, MPI_COMM_WORLD) != MPI_SUCCESS){
            error("Something went wrong sending the routing table!\n");
        }

        return 0;
    }

    virtual int handshakeHandler(){
        int sz;
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, DFF_GROUP_NAME_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &sz);
        char* buff = new char [sz];
        MPI_Recv(buff, sz, MPI_BYTE, status.MPI_SOURCE, DFF_GROUP_NAME_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<int> reachableDestinations;
        for(const auto& [key, value] : this->routingTable) reachableDestinations.push_back(key);

        return this->sendRoutingTable(status.MPI_SOURCE, reachableDestinations);
    }

    virtual void registerEOS(int rank){
        neos++;
    }

    virtual void forward(message_t* task, int){
        if (task->chid == -1) ff_send_out(task);
        else ff_send_out_to(task, this->routingTable[task->chid]); // assume the routing table is consistent WARNING!!!
    }


public:
    ff_dreceiverMPI(size_t input_channels, std::map<int, int> routingTable = {std::make_pair(0,0)}, int coreid=-1)
		: input_channels(input_channels), routingTable(routingTable), coreid(coreid) {}

    int svc_init() {
  		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
        

        for(size_t i = 0; i < input_channels; i++)
            handshakeHandler();

        return 0;
    }

    /* 
        Here i should not care of input type nor input data since they come from a socket listener.
        Everything will be handled inside a while true in the body of this node where data is pulled from network
    */
    message_t *svc(message_t* task) {
        MPI_Status status;
        while(neos < input_channels){
            
            int headersLen;
            MPI_Probe(MPI_ANY_SOURCE, DFF_HEADER_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_LONG, &headersLen);
            long headers[headersLen];

            if (MPI_Recv(headers, headersLen, MPI_LONG, status.MPI_SOURCE, DFF_HEADER_TAG, MPI_COMM_WORLD, &status) != MPI_SUCCESS)
                error("Error on Recv Receiver primo in alto\n");
            
            assert(headers[0]*3+1 == headersLen);
            if (headers[0] == 1){
                 size_t sz = headers[3];

                if (sz == 0){
                    registerEOS(status.MPI_SOURCE);
                    continue;
                }
                char* buff = new char[sz];
                if (MPI_Recv(buff,sz,MPI_BYTE, status.MPI_SOURCE, DFF_TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS)
                    error("Error on Recv Receiver Payload\n");
                
                message_t* out = new message_t(buff, sz, true);
                out->sender = headers[1];
                out->chid   = headers[2];

                this->forward(out, status.MPI_SOURCE);
            } else {
                int size;
                MPI_Status localStatus;
                MPI_Probe(status.MPI_SOURCE, DFF_TASK_TAG, MPI_COMM_WORLD, &localStatus);
                MPI_Get_count(&localStatus, MPI_BYTE, &size);
                char* buff = new char[size]; // this can be reused!! 
                MPI_Recv(buff, size, MPI_BYTE, localStatus.MPI_SOURCE, DFF_TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                size_t head = 0;
                for (size_t i = 0; i < (size_t)headers[0]; i++){
                    size_t sz = headers[3*i+3];
                    if (sz == 0){
                        registerEOS(status.MPI_SOURCE);
                        assert(i+1 == (size_t)headers[0]);
                        break;
                    }
                    char* outBuff = new char[sz];
                    memcpy(outBuff, buff+head, sz);
                    head += sz;
                    message_t* out = new message_t(outBuff, sz, true);
                    out->sender = headers[3*i+1];
                    out->chid = headers[3*i+2];

                    this->forward(out, status.MPI_SOURCE);
                }
                delete [] buff;
            }
            
        }
        
        return this->EOS;
    }

protected:
    size_t neos = 0;
    size_t input_channels;
    std::map<int, int> routingTable;
	int coreid;
};



class ff_dreceiverHMPI : public ff_dreceiverMPI {
    std::vector<int> internalDestinations;
    std::set<std::string> internalGroupNames;
    std::set<int> internalRanks;
    size_t internalNEos = 0, externalNEos = 0;
    int next_rr_destination = 0;


    virtual void registerEOS(int rank){
        neos++;
  
        if (!internalRanks.contains(rank)){
            if (++externalNEos == (input_channels-internalRanks.size()))
				for(size_t i = 0; i < this->get_num_outchannels()-1; i++) ff_send_out_to(this->EOS, i);
        } else
			if (++internalNEos == internalRanks.size())
				ff_send_out_to(this->EOS, this->get_num_outchannels()-1);
    }

    virtual int handshakeHandler(){
        int sz;
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, DFF_GROUP_NAME_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &sz);
        char* buff = new char [sz];
        MPI_Recv(buff, sz, MPI_BYTE, status.MPI_SOURCE, DFF_GROUP_NAME_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // the connection is internal!
        if (internalGroupNames.contains(std::string(buff, sz))) {
            internalRanks.insert(status.MPI_SOURCE);
            return this->sendRoutingTable(status.MPI_SOURCE, internalDestinations);
        }

        std::vector<int> reachableDestinations;
        for(const auto& [key, value] : this->routingTable) reachableDestinations.push_back(key);

        return this->sendRoutingTable(status.MPI_SOURCE, reachableDestinations);
    }

    void forward(message_t* task, int rank){
        if (internalRanks.contains(rank)) ff_send_out_to(task, this->get_num_outchannels()-1);
        else if (task->chid != -1) ff_send_out_to(task, this->routingTable[task->chid]);
        else {
            ff_send_out_to(task, next_rr_destination);
            next_rr_destination = ((next_rr_destination + 1) % (this->get_num_outchannels()-1));
        }
    }

public:
    ff_dreceiverHMPI(size_t input_channels, std::map<int, int> routingTable = {std::make_pair(0,0)}, std::vector<int> internalRoutingTable = {0}, std::set<std::string> internalGroups = {}, int coreid=-1)
		: ff_dreceiverMPI(input_channels, routingTable, coreid), internalDestinations(internalRoutingTable), internalGroupNames(internalGroups)  {}

};




/** versione Ondemand */
class ff_dreceiverMPIOD: public ff_dreceiverMPI { 
public:
    ff_dreceiverMPIOD(size_t input_channels, std::map<int, int> routingTable = {std::make_pair(0,0)}, int coreid=-1)
		: ff_dreceiverMPI(input_channels, routingTable, coreid) {}
    /* 
        Here i should not care of input type nor input data since they come from a socket listener.
        Everything will be handled inside a while true in the body of this node where data is pulled from network
    */
    message_t *svc(message_t* task) {
        MPI_Request tmpAckReq;
        MPI_Status status;
        long header[3];
        while(neos < input_channels){
            MPI_Recv(header, 3, MPI_LONG, MPI_ANY_SOURCE, DFF_HEADER_TAG, MPI_COMM_WORLD, &status);

            size_t sz = header[0];

            if (sz == 0){
                neos++;
                continue;
            }

            char* buff = new char [sz];
			assert(buff);

            MPI_Recv(buff,sz,MPI_BYTE, status.MPI_SOURCE, DFF_TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            message_t* out = new message_t(buff, sz, true);
			assert(out);
			out->sender = header[1];
			out->chid   = header[2];

            //std::cout << "received something from " << sender << " directed to " << chid << std::endl;
            if (out->chid != -1)
                ff_send_out_to(out, this->routingTable[out->chid]); // assume the routing table is consistent WARNING!!!
            else
                ff_send_out(out);

            MPI_Isend(&ACK, sizeof(ack_t), MPI_BYTE, status.MPI_SOURCE, DFF_ACK_TAG, MPI_COMM_WORLD, &tmpAckReq);
            MPI_Request_free(&tmpAckReq);
        }
        
        return this->EOS;
    }

private:
    ack_t ACK;
};

#endif
