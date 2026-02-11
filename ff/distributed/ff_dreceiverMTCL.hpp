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

    template <typename T>
    static inline T NetworkToHostByteOrder(T value) {
        static_assert(std::is_integral<T>::value, "T must be an integral type.");

        if constexpr (sizeof(T) == 2)  // 16-bit values
            return ntohs(static_cast<uint16_t>(value));
        else if constexpr (sizeof(T) == 4)  // 32-bit values
            return ntohl(static_cast<uint32_t>(value));
         else if constexpr (sizeof(T) == 8)  // 64-bit values (no standard functions, so implemented manually)
            return be64toh(static_cast<uint64_t>(value)); 
         else 
            return value;
    }

    inline int handleBatch(char*& inputBuffer, size_t size, MTCL::HandleUser& h){

#ifdef DFF_ACK
        if (h.send(&ACK, sizeof(ack_t)) < 0){
            error("dreceiver: Error sending back ack");
        }
#endif

        int elementsInBatch = ntohl(*reinterpret_cast<int*>(inputBuffer + size - sizeof(int)));

        if (elementsInBatch == 1){
            char* headerBuffer = inputBuffer + size - headerSize - sizeof(int);
            addr_t src = NetworkToHostByteOrder(*reinterpret_cast<addr_t*>(headerBuffer+sizeof(addr_t)));        
                        
            if (*reinterpret_cast<sizeDFF_t*>(headerBuffer+2*sizeof(addr_t)+sizeof(ChannelType))){
                message2_t* out = MessageAllocator::allocateMessage();
                out->size = NetworkToHostByteOrder(*reinterpret_cast<sizeDFF_t*>(headerBuffer+2*sizeof(addr_t)+sizeof(ChannelType)));
                out->dest = NetworkToHostByteOrder(*reinterpret_cast<addr_t*>(headerBuffer));
                out->type = *reinterpret_cast<ChannelType*>(headerBuffer+2*sizeof(addr_t));
                if (out->size > SINGLE_SEND_SIZE_THRESHOLD){
                    out->data = (char*)malloc(out->size);
                    h.receive(out->data, out->size);
                } else{
                    out->data = inputBuffer;
                    inputBuffer = nullptr;
                }

                out->cleanup = true;
			    out->src = src;
                out->locality = ChannelLocality::REMOTE;
                ff_send_out(out);
                if (!inputBuffer)
                    inputBuffer = (char*)malloc(size); 

                return 0;
            }

            if (src != -1){
                ff_send_out(MessageAllocator::make_logical_EOS(src));
                return 0;
            }
            
            return -1;

        } else {
            char* payload_sliding_ptr = inputBuffer;
            char* headerBuffer_sliding_ptr = inputBuffer + size - elementsInBatch*headerSize - sizeof(int);
            for(int i = 0; i < elementsInBatch; i++){
                addr_t src = NetworkToHostByteOrder(*reinterpret_cast<addr_t*>(headerBuffer_sliding_ptr+sizeof(addr_t)));

                if (*reinterpret_cast<sizeDFF_t*>(headerBuffer_sliding_ptr+2*sizeof(addr_t)+sizeof(ChannelType))){
                    message2_t* out = MessageAllocator::allocateMessage();
                    out->size = NetworkToHostByteOrder(*reinterpret_cast<sizeDFF_t*>(headerBuffer_sliding_ptr+2*sizeof(addr_t)+sizeof(ChannelType))); 

                    out->data = (char*) malloc(out->size);
                    std::memcpy(out->data, payload_sliding_ptr, out->size);
                    payload_sliding_ptr += out->size; // advance the sliding ptr
                    out->cleanup = true;
                
                    out->dest = NetworkToHostByteOrder(*reinterpret_cast<addr_t*>(headerBuffer_sliding_ptr));
                    out->src   = src;
                    out->type = *reinterpret_cast<ChannelType*>(headerBuffer_sliding_ptr+2*sizeof(addr_t));
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

        }
        
        return 0;
    }

    

public:
    ff_dreceiverMTCL2(std::string& acceptAddr, size_t input_channels, bool straightGroup = false)
		: input_channels(input_channels), acceptAddr(acceptAddr){ }

    int svc_init() {
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
        
        std::vector<MTCL::HandleUser> handles;
        char* buffers[2] = {nullptr, nullptr};
        size_t buffSizes[2] = {0, 0};
        size_t seq_nr = 0;
        struct PendingOp {
            bool active = false;
            MTCL::Request request;
            size_t size = 0;
            size_t handleIdx = 0;
            long timestamp = 0;
        };
        PendingOp pending[2];
        size_t lastIdxProbed = 0;

        while (ff::termination_counter > 0) {
            // Accept new connections (non-blocking)
            while (handles.size() < input_channels) {
                auto handle = MTCL::Manager::getNext(std::chrono::microseconds(0));
                if (handle.isValid() && handle.isNewConnection()) {
                    handles.push_back(std::move(handle));
                }
            }

            // Count active operations
            int activeCount = (pending[0].active ? 1 : 0) + (pending[1].active ? 1 : 0);
            
            // Try to start new receives if we have free slots
            if (activeCount < 2) {
                size_t idx_ = 0;
                for (; idx_ < handles.size() && activeCount < 2; ++idx_) {
                    size_t idx = (lastIdxProbed + idx_) % handles.size();
                    auto& h = handles[idx];

                    if (h.isClosed().first) continue; // if the the handle is closed in read (first) just skip this handle

                    // Check if data is available (non-blocking)
                    size_t sz = 0;
                    int probeResult = h.probe(sz, false);
                    
                    if (probeResult < 0) {
                        if (errno == EWOULDBLOCK) {
                            continue; // No data available yet
                        } else {
                            // Handle error - mark handle as invalid
                            h.close();
                            continue;
                        }
                    }
                    
                    if (sz == 0) {
                        
                        // Cancel any pending operations for this handle
                        for (int i = 0; i < 2; ++i) {
                            if (pending[i].active && pending[i].handleIdx == idx) {
                                
                                // handle pending incoming batch
                                MTCL::wait(pending[i].request);
                                handleBatch(buffers[i], pending[i].size, handles[pending[i].handleIdx]);
                                
                                pending[i].active = false;
                                activeCount--;
                            }
                        }

                        h.close(); // Close connection
                        continue;
                    }

                    // Find a free slot
                    int slot = -1;
                    for (int i = 0; i < 2; ++i) {
                        if (!pending[i].active) {
                            slot = i;
                            break;
                        }
                    }
                    
                    if (slot == -1) continue; // No free slots (shouldn't happen)

                    // Ensure buffer capacity
                    if (sz > buffSizes[slot]) {
                        buffers[slot] = static_cast<char*>(realloc(buffers[slot], sz));
                        buffSizes[slot] = sz;
                    }

                    // Start non-blocking receive
                    h.ireceive(buffers[slot], sz, pending[slot].request);
                    
               
                    pending[slot].active = true;
                    pending[slot].size = sz;
                    pending[slot].handleIdx = idx;
                    pending[slot].timestamp = seq_nr++;
                    activeCount++;
                }
                lastIdxProbed = (lastIdxProbed + idx_) % handles.size();
            }

            // Check for completed operations
            if (pending[0].active && pending[1].active && pending[0].handleIdx == pending[1].handleIdx){
                if (pending[0].timestamp < pending[1].timestamp){
                    MTCL::wait(pending[0].request);
                    handleBatch(buffers[0], pending[0].size, handles[pending[0].handleIdx]);
                    pending[0].active = false;
                
                } else {
                    // processa 1;
                    MTCL::wait(pending[1].request);
                    handleBatch(buffers[1], pending[1].size, handles[pending[1].handleIdx]);
                    pending[1].active = false;
                    
                }
            } else
            for (int i = 0; i < 2; ++i) {
                if (pending[i].active && MTCL::test(pending[i].request)) {
                    handleBatch(buffers[i], pending[i].size, handles[pending[i].handleIdx]);
                    pending[i].active = false;
                }
            }
        }

        for(int i  = 0; i < 2; i++)
            free(buffers[i]);

        return this->EOS;
    }

protected:
    size_t input_channels;
    std::string& acceptAddr;	
    ack_t ACK;
};

} // namespace 
#endif
