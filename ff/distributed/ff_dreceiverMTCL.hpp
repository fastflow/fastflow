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

#ifndef FF_DRECEIVER_H
#define FF_DRECEIVER_H


#include <iostream>
#include <sstream>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_dgroups.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <arpa/inet.h>
#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>

#include "MTCL/include/mtcl.hpp"


namespace ff {

class ff_dreceiverMTCL: public ff_monode_t<message_t> { 
protected:
    std::map<size_t, ChannelType> connID2ChannelType;
    static const size_t headerSize = sizeof(int)+sizeof(int)+sizeof(size_t);

    virtual int handshakeHandler(MTCL::HandleUser& h){
        // ricevo l'handshake e mi salvo che tipo di connessione è
        size_t size;
       
        if(h.probe(size, true) < 0){
            error("dreceiver handshakehandler: error probe");
            return -1;
        }
        char* buff = new char[size];
        assert(buff);

        if (h.receive(buff, size) < 0){
            error("dreceiver handshakehandler: error receving");
            return -1;
        };

        connID2ChannelType[h.getID()] = *reinterpret_cast<ChannelType*>(buff);
        delete [] buff;
        return 0; //this->sendRoutingTable(sck, reachableDestinations);
    }

    virtual void registerLogicalEOS(int sender){
        for(size_t i = 0; i < this->get_num_outchannels(); i++)
                ff_send_out_to(new message_t(sender, i), i);
    }

    virtual void registerEOS(size_t connID){
        this->neos++;
    }

    virtual void forward(message_t* task, size_t){
        if (task->chid == -1) ff_send_out(task);
        else ff_send_out_to(task, this->routingTable[task->chid]); // assume the routing table is consistent WARNING!!!
    }

    virtual int handleBatch(MTCL::HandleUser& h){
        ChannelType t = connID2ChannelType[h.getID()];
        size_t size;

        if (h.probe(size) < 0){ // porbe header
            error("dreceiver probe header error");
            return -1;
        } 

        if (size == 0){
            h.close();
            return -1;
        }
        char* headerBuffer = new char[size];
        if (h.receive(headerBuffer, size) < 0){
            error("dreceiver receive header error");
            return -1;
        }
        size_t elementsInBatch = size/headerSize;
        bool payload = false;
        for(size_t i = 0; i < elementsInBatch; i++) 
            if (*reinterpret_cast<size_t*>(headerBuffer+i*headerSize+2*sizeof(int))) {
                payload = true;
                break;
            }
        size_t payloadSize = 0;
        char* payloadBuffer = nullptr;
        if (payload){
            if (h.probe(payloadSize) < 0){
                error("dreiceiver probe payload error");
                return -1;
            };
            payloadBuffer = new char[payloadSize];
            if (h.receive(payloadBuffer, payloadSize) < 0){
                error("dreceiver receive payload error");
                return -1;
            }
        }

        if (h.send(&ACK, sizeof(ack_t)) < 0){
            error("dreceiver: Error sending back ack");
        }

        if (elementsInBatch == 1){
            int sender = ntohl(*reinterpret_cast<int*>(headerBuffer));
            int chid = ntohl(*reinterpret_cast<int*>(headerBuffer+sizeof(int)));
            size_t sz = be64toh(*reinterpret_cast<size_t*>(headerBuffer+2*sizeof(int)));
            assert(sz == payloadSize);

            if (sz){
                message_t* out = new message_t(payloadBuffer, sz, true);
			
                out->feedback = t == ChannelType::FBK;
			    out->sender = sender;
			    out->chid   = chid;
                this->forward(out, h.getID());
                delete [] headerBuffer;
                return 0;
            }

            if (chid == -2){
                registerLogicalEOS(sender);
                delete [] headerBuffer;
                return 0;
            }

            registerEOS(h.getID());
            delete [] headerBuffer;
            return -1;

        } else {
            char* payload_sliding_ptr = payloadBuffer;
            char* headerBuffer_sliding_ptr = headerBuffer;
            for(size_t i = 0; i < elementsInBatch; i++){
                int sender = ntohl(*reinterpret_cast<int*>(headerBuffer_sliding_ptr));
                int chid = ntohl(*reinterpret_cast<int*>(headerBuffer_sliding_ptr+sizeof(int)));
                size_t sz = be64toh(*reinterpret_cast<size_t*>(headerBuffer_sliding_ptr+2*sizeof(int)));

                if (sz){
                    char* _buf = new char[sz];
                    memcpy(_buf, payload_sliding_ptr, sz);
                    payload_sliding_ptr += sz; // advance the sliding ptr
                    message_t* out = new message_t(_buf, sz, true);
                
                    out->feedback = t == ChannelType::FBK;
                    out->sender = sender;
                    out->chid   = chid;
                    this->forward(out, h.getID());
                    headerBuffer_sliding_ptr += headerSize;
                    continue;
                }

                if (chid == -2){
                    registerLogicalEOS(sender);
                    headerBuffer_sliding_ptr += headerSize;
                    continue;
                }

                registerEOS(h.getID());

                headerBuffer_sliding_ptr += headerSize;
            }

            if (payloadBuffer)
                delete [] payloadBuffer; 
        }

        delete [] headerBuffer;
        
        return 0;
    }

    

public:
    ff_dreceiverMTCL(ff_endpoint acceptAddr, size_t input_channels, std::map<int, int> routingTable = {std::make_pair(0,0)}, int coreid=-1)
		: input_channels(input_channels), acceptAddr(acceptAddr), routingTable(routingTable), coreid(coreid) {}

    int svc_init() {
  		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
        if (MTCL::Manager::listen(acceptAddr.address) < 0){
            error("Error in MTCL listen in dreceiver");
            return -1;
        }
        
        std::cout << "Receiver initialized!\n";

        return 0;
    }

    /* 
        Here i should not care of input type nor input data since they come from a socket listener.
        Everything will be handled inside a while true in the body of this node where data is pulled from network
    */
    message_t *svc(message_t* task) {

        while(neos < input_channels){
            auto handle = MTCL::Manager::getNext();
            if(handle.isNewConnection())
                this->handshakeHandler(handle);
             else 
                this->handleBatch(handle);
        }
		
        return this->EOS;
    }

protected:
    size_t neos = 0;
    size_t input_channels;
    ff_endpoint acceptAddr;	
    std::map<int, int> routingTable;
	int coreid;
    ack_t ACK;
};


class ff_dreceiverHMTCL : public ff_dreceiverMTCL {

    //std::map<int, bool> isInternalConnection;
    size_t internalNEos = 0, externalNEos = 0;
    long next_rr_destination = 0;

    void registerLogicalEOS(int sender){
        for(size_t i = 0; i < this->get_num_outchannels()-1; i++)
                ff_send_out_to(new message_t(sender, i), i);
    }

    void registerEOS(size_t connID){
        neos++;
        size_t internalConn = std::count_if(std::begin(connID2ChannelType),
                                            std::end  (connID2ChannelType),
                                            [](std::pair<int, ChannelType> const &p) {return p.second == ChannelType::INT;});

        if (connID2ChannelType[connID] != ChannelType::INT){
            // logical EOS!!
            /*for(int i = 0; i < this->get_num_outchannels()-1; i++)
                ff_send_out(new message_t(0,0), i);*/

            if (++externalNEos == (connID2ChannelType.size()-internalConn))
				for(size_t i = 0; i < this->get_num_outchannels()-1; i++) ff_send_out_to(this->EOS, i);
        } else{
            /// logical EOS!
            //ff_send_out_to(new message_t(0,0), this->get_num_outchannels()-1); 
			
            if (++internalNEos == internalConn)
				ff_send_out_to(this->EOS, this->get_num_outchannels()-1);
        }
        
        
    }

    void forward(message_t* task, size_t connID){
        if (connID2ChannelType[connID] == ChannelType::INT) ff_send_out_to(task, this->get_num_outchannels()-1);
        else if (task->chid != -1) ff_send_out_to(task, this->routingTable[task->chid]);
        else {
            ff_send_out_to(task, next_rr_destination);
            next_rr_destination = (next_rr_destination + 1) % (this->get_num_outchannels()-1);
        }
    }

public:
    ff_dreceiverHMTCL(ff_endpoint acceptAddr, size_t input_channels, std::map<int, int> routingTable = {{0,0}}, int coreid=-1) 
    : ff_dreceiverMTCL(acceptAddr, input_channels, routingTable, coreid){

    }

};

} // namespace 
#endif
