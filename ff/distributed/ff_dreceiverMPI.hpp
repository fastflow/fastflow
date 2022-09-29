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

#define MPI_SLEEPTIME 200000 //nanoseconds

#ifdef BLOCKING_MODE
#define MPI_RECV_CALL MPI_Recv_NB
#define MPI_PROBE_CALL MPI_Probe_NB
#else
#define MPI_RECV_CALL MPI_Recv
#define MPI_PROBE_CALL MPI_Probe
#endif

namespace ff {

    const struct timespec mpiSleepTime = {0, MPI_SLEEPTIME};

    inline int MPI_Recv_NB(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status){
        MPI_Request r;
        int returnValue = MPI_Irecv(buf, count, datatype, source, tag, comm, &r);
        if (returnValue != MPI_SUCCESS) return returnValue;
        int flag = 0;
        while(!flag){
            returnValue = MPI_Test(&r, &flag, status);
            nanosleep(&ff::mpiSleepTime, NULL);
        }
        return returnValue;
    }

    inline int MPI_Probe_NB(int source, int tag, MPI_Comm comm, MPI_Status * status){
        int flag = 0, returnValue;
        while(!flag) {
            returnValue = MPI_Iprobe(source, tag, comm, &flag, status);
            nanosleep(&ff::mpiSleepTime, NULL);
        }
        return returnValue;
    }


class ff_dreceiverMPI: public ff_monode_t<message_t> { 
protected:

    /*static int sendRoutingTable(const int rank, const std::vector<int>& dest){
        dataBuffer buff; std::ostream oss(&buff);
		cereal::PortableBinaryOutputArchive oarchive(oss);
		oarchive << dest;

        if (MPI_Send(buff.getPtr(), buff.getLen(), MPI_BYTE, rank, DFF_ROUTING_TABLE_TAG, MPI_COMM_WORLD) != MPI_SUCCESS){
            error("Something went wrong sending the routing table!\n");
        }

        return 0;
    }*/

    virtual int handshakeHandler(){
        int sz;
        ChannelType ct;
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, DFF_GROUP_NAME_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &sz);
        char* buff = new char [sz];
        MPI_Recv(buff, sz, MPI_BYTE, status.MPI_SOURCE, DFF_GROUP_NAME_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&ct, sizeof(ct), MPI_BYTE, status.MPI_SOURCE, DFF_CHANNEL_TYPE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        rank2ChannelType[status.MPI_SOURCE] = ct;

        return 0;
    }

    virtual void registerLogicalEOS(int sender){
        for(size_t i = 0; i < this->get_num_outchannels(); i++)
            ff_send_out_to(new message_t(sender, i), i);
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
            MPI_PROBE_CALL(MPI_ANY_SOURCE, DFF_HEADER_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_LONG, &headersLen);
            long headers[headersLen];

            if (MPI_RECV_CALL(headers, headersLen, MPI_LONG, status.MPI_SOURCE, DFF_HEADER_TAG, MPI_COMM_WORLD, &status) != MPI_SUCCESS)
                error("Error on Recv Receiver\n");
            bool feedback = ChannelType::FBK == rank2ChannelType[status.MPI_SOURCE];
            assert(headers[0]*3+1 == headersLen);
            if (headers[0] == 1){
                 size_t sz = headers[3];

                if (sz == 0){
                    if (headers[2] == -2){
                        registerLogicalEOS(headers[1]);
                        continue;
                    }
                    registerEOS(status.MPI_SOURCE);
                    continue;
                }
                char* buff = new char[sz];
                if (MPI_RECV_CALL(buff,sz,MPI_BYTE, status.MPI_SOURCE, DFF_TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS)
                    error("Error on Recv Receiver Payload\n");
                
                message_t* out = new message_t(buff, sz, true);
                out->sender = headers[1];
                out->chid   = headers[2];
                out->feedback = feedback;

                this->forward(out, status.MPI_SOURCE);
            } else {
                int size;
                MPI_Status localStatus;
                MPI_PROBE_CALL(status.MPI_SOURCE, DFF_TASK_TAG, MPI_COMM_WORLD, &localStatus);
                MPI_Get_count(&localStatus, MPI_BYTE, &size);
                char* buff = new char[size]; // this can be reused!! 
                MPI_RECV_CALL(buff, size, MPI_BYTE, localStatus.MPI_SOURCE, DFF_TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                size_t head = 0;
                for (size_t i = 0; i < (size_t)headers[0]; i++){
                    size_t sz = headers[3*i+3];
                    if (sz == 0){
                        if (headers[3*i+2] == -2){
                            registerLogicalEOS(headers[3*i+1]);
                            continue;
                        }
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
                    out->feedback = feedback;

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
    std::map<int, ChannelType> rank2ChannelType;
	int coreid;
};



class ff_dreceiverHMPI : public ff_dreceiverMPI {
    size_t internalNEos = 0, externalNEos = 0;
    int next_rr_destination = 0;

    virtual void registerLogicalEOS(int sender){
        for(size_t i = 0; i < this->get_num_outchannels()-1; i++)
            ff_send_out_to(new message_t(sender, i), i);
    }

    virtual void registerEOS(int rank){
        neos++;

        size_t internalConn = std::count_if(std::begin(rank2ChannelType),
                                            std::end  (rank2ChannelType),
                                            [](std::pair<int, ChannelType> const &p) {return p.second == ChannelType::INT;});

  
        if (rank2ChannelType[rank] != ChannelType::INT){
            if (++externalNEos == (rank2ChannelType.size()-internalConn))
				for(size_t i = 0; i < this->get_num_outchannels()-1; i++) ff_send_out_to(this->EOS, i);
        } else
			if (++internalNEos == internalConn)
				ff_send_out_to(this->EOS, this->get_num_outchannels()-1);
    }

    void forward(message_t* task, int rank){
        if (rank2ChannelType[rank] == ChannelType::INT) ff_send_out_to(task, this->get_num_outchannels()-1);
        else if (task->chid != -1) ff_send_out_to(task, this->routingTable[task->chid]);
        else {
            ff_send_out_to(task, next_rr_destination);
            next_rr_destination = ((next_rr_destination + 1) % (this->get_num_outchannels()-1));
        }
    }

public:
    ff_dreceiverHMPI(size_t input_channels, std::map<int, int> routingTable = {std::make_pair(0,0)}, int coreid=-1)
		: ff_dreceiverMPI(input_channels, routingTable, coreid)  {}

};

} // namespace
#endif
