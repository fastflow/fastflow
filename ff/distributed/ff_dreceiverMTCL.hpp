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
#include <ff/distributed/ff_ddefines.hpp>
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

class ff_dreceiverMTCL2: public ff_monode_t<message2_t> { 
protected:
    static const size_t headerSize = sizeof(int)+sizeof(int)+sizeof(ChannelType)+sizeof(size_t);

    inline int handleBatch(MTCL::HandleUser& h){
        size_t size;

        if (h.probe(size) < 0){ // porbe header
            error("dreceiver probe header error");
            return -1;
        } 

        if (size == 0){
            h.close();
            return -1;
        }
        size_t elementsInBatch = size/headerSize;

        if (elementsInBatch > headerBufferEntries){
            headerBuffer = (char*)realloc(headerBuffer, elementsInBatch*headerSize);
            assert(headerBuffer);
            headerBufferEntries = elementsInBatch;
        }

        if (h.receive(headerBuffer, size) < 0){
            error("dreceiver receive header error");
            return -1;
        }
    
        if (h.send(&ACK, sizeof(ack_t)) < 0){
            error("dreceiver: Error sending back ack");
        }
     
        bool payload = false;
        for(size_t i = 0; i < elementsInBatch; i++) 
            if (*reinterpret_cast<size_t*>(headerBuffer+i*headerSize+2*sizeof(int)+sizeof(ChannelType))) {
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
            assert(payloadBuffer);
            if (h.receive(payloadBuffer, payloadSize) < 0){
                error("dreceiver receive payload error");
                return -1;
            }
        }

        if (elementsInBatch == 1){
            int dest = ntohl(*reinterpret_cast<int*>(headerBuffer));
            int src = ntohl(*reinterpret_cast<int*>(headerBuffer+sizeof(int)));
            ChannelType t = *reinterpret_cast<ChannelType*>(headerBuffer+2*sizeof(int));
            size_t sz = be64toh(*reinterpret_cast<size_t*>(headerBuffer+2*sizeof(int)+sizeof(ChannelType)));
            assert(sz == payloadSize);
                        
            if (sz){
                message2_t* out = MessageAllocator::allocateMessage();
                out->data = payloadBuffer; out->size = sz; out->cleanup = true;

			    out->src = src;
			    out->dest   = dest;
                out->type = t;
                out->locality = ChannelLocality::REMOTE;
                ff_send_out(out);
                return 0;
            }

            if (src != -1){
                ff_send_out(MessageAllocator::make_logical_EOS(src));
                return 0;
            }
            
            return -1;

        } else {
            char* payload_sliding_ptr = payloadBuffer;
            char* headerBuffer_sliding_ptr = headerBuffer;
            for(size_t i = 0; i < elementsInBatch; i++){
                int dest = ntohl(*reinterpret_cast<int*>(headerBuffer_sliding_ptr));
                int src = ntohl(*reinterpret_cast<int*>(headerBuffer_sliding_ptr+sizeof(int)));
                ChannelType t = *reinterpret_cast<ChannelType*>(headerBuffer_sliding_ptr+2*sizeof(int));
                size_t sz = be64toh(*reinterpret_cast<size_t*>(headerBuffer_sliding_ptr+2*sizeof(int)+sizeof(ChannelType)));

                if (sz){
                    char* _buf = new char[sz];
                    memcpy(_buf, payload_sliding_ptr, sz);
                    payload_sliding_ptr += sz; // advance the sliding ptr
                    message2_t* out = MessageAllocator::allocateMessage();
                    out->data = _buf; out->size = sz; out->cleanup = true;
                
                    out->dest = dest;
                    out->src   = src;
                    out->type = t;
                    ff_send_out(out);
                    headerBuffer_sliding_ptr += headerSize;
                    continue;
                }

                if (src != -1){
                    ff_send_out(MessageAllocator::make_logical_EOS(src));
                    headerBuffer_sliding_ptr += headerSize;
                    continue;
                }

                headerBuffer_sliding_ptr += headerSize;
            }

            if (payloadBuffer)
                delete [] payloadBuffer; 
        }
        
        return 0;
    }

    

public:
    ff_dreceiverMTCL2(std::string& acceptAddr, size_t input_channels, int coreid=-1)
		: input_channels(input_channels), acceptAddr(acceptAddr), coreid(coreid), headerBufferEntries(1) {
            headerBuffer = (char*)malloc(headerSize*headerBufferEntries);
            assert(headerBuffer);
        }

    int svc_init() {
  		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
        if (MTCL::Manager::listen(acceptAddr) < 0){
            error("Error in MTCL listen in dreceiver");
            return -1;
        }
        return 0;
    }

    /* 
        Here i should not care of input type nor input data since they come from a socket listener.
        Everything will be handled inside a while true in the body of this node where data is pulled from network
    */
    message2_t *svc(message2_t* task) {

        while(neos < input_channels){
            auto handle = MTCL::Manager::getNext(std::chrono::microseconds(RECEIVER_POLL_TIMEOUT));
            if (handle.isValid()) 
                this->handleBatch(handle);

            if (ff::termination_counter <= 0)
                break;
            
        }
        return this->EOS;
    }

protected:
    size_t neos = 0;
    size_t input_channels;
    std::string& acceptAddr;	
	int coreid;
    ack_t ACK;
    char* headerBuffer = nullptr;
    size_t headerBufferEntries = 1;
};

} // namespace 
#endif
