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

        return 0;
    }

    /* 
        Here i should not care of input type nor input data since they come from a socket listener.
        Everything will be handled inside a while true in the body of this node where data is pulled from network
    */
    message_t *svc(message_t* task) {
        MPI_Status status;
        int header[3];
        while(neos < input_channels){
            MPI_Recv(header, 3, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == DFF_ROUTING_TABLE_TAG){
                sendRoutingTable(status.MPI_SOURCE);
                continue;
            }

            int sz = header[0];

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

            ff_send_out_to(out, this->routingTable[out->chid]); // assume the routing table is consistent WARNING!!!
        }
        
        return this->EOS;
    }

private:
    size_t neos = 0;
    size_t input_channels;
    std::map<int, int> routingTable;
	int coreid;
};

/** versione Ondemand */


class ff_dreceiverMPIOD: public ff_monode_t<message_t> { 
private:

    int sendRoutingTable(int rank){
        dataBuffer buff; std::ostream oss(&buff);
		cereal::PortableBinaryOutputArchive oarchive(oss);
        std::vector<int> reachableDestinations;

        for(auto const& p : this->routingTable) reachableDestinations.push_back(p.first);

		oarchive << reachableDestinations;

        if (MPI_Send(buff.getPtr(), buff.getLen(), MPI_BYTE, rank, DFF_ROUTING_TABLE_TAG, MPI_COMM_WORLD) != MPI_SUCCESS)
            return -1;

        return 0;
    }

public:
    ff_dreceiverMPIOD(size_t input_channels, std::map<int, int> routingTable = {std::make_pair(0,0)}, bool onDemand_ = false, int coreid=-1)
		: input_channels(input_channels), routingTable(routingTable), onDemand(onDemand_), coreid(coreid) {}

    int svc_init() {
  		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);

        return 0;
    }

    /* 
        Here i should not care of input type nor input data since they come from a socket listener.
        Everything will be handled inside a while true in the body of this node where data is pulled from network
    */
    message_t *svc(message_t* task) {
        MPI_Request tmpAckReq;
        MPI_Status status;
        int header[3];
        while(neos < input_channels){
            MPI_Recv(header, 3, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == DFF_ROUTING_TABLE_TAG){
                sendRoutingTable(status.MPI_SOURCE);
                continue;
            }

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

            if (onDemand){
                MPI_Isend(&ACK, sizeof(ack_t), MPI_BYTE, status.MPI_SOURCE, DFF_ACK_TAG, MPI_COMM_WORLD, &tmpAckReq);
                MPI_Request_free(&tmpAckReq);
            }
        }
        
        return this->EOS;
    }

private:
    size_t neos = 0;
    size_t input_channels;
    std::map<int, int> routingTable;
    bool onDemand;
	int coreid;
    ack_t ACK;
};

#endif
