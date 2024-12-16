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

#ifndef FF_DSENDER_H
#define FF_DSENDER_H

#include <iostream>
#include <map>
#include <random>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_messageAllocator.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <netdb.h>
#include <cmath>
#include <thread>

#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>

#include "MTCL/include/mtcl.hpp"

namespace ff {

class ff_dsenderMTCL2;

class uBuffer_i {
    friend class ff_dsenderMTCL2;
protected:
    ff_dsenderMTCL2* parent;
    static const size_t headerSize = sizeof(int)+sizeof(int)+sizeof(ChannelType)+sizeof(size_t);
    size_t maxSize, capacity;
    char *base_ptr = nullptr, *current_ptr = nullptr;
    char *headers_buffer_ptr = nullptr;
    int flush();
    int flush(void*, size_t);
    
    inline static void pushHeader_(char* buffAddr, int dest, int src, ChannelType t, size_t size_){
        *reinterpret_cast<int*>(buffAddr) = dest;
        *reinterpret_cast<int*>(buffAddr+sizeof(int)) = src;
        *reinterpret_cast<ChannelType*>(buffAddr+2*sizeof(int)) = t;
        *reinterpret_cast<size_t*>(buffAddr+2*sizeof(int)+sizeof(ChannelType)) = size_;
    }

    inline void pushHeader(int dest, int src, ChannelType t, size_t size_){
        char* offset_ptr = headers_buffer_ptr + this->size*headerSize;
        *reinterpret_cast<int*>(offset_ptr) = dest;
        *reinterpret_cast<int*>(offset_ptr+sizeof(int)) = src;
        *reinterpret_cast<ChannelType*>(offset_ptr+2*sizeof(int)) = t;
        *reinterpret_cast<size_t*>(offset_ptr+2*sizeof(int)+sizeof(ChannelType)) = size_;
    }

public:
    uBuffer_i(ff_dsenderMTCL2* parent, size_t connID, size_t batchSize, size_t taskSizeHint = 512) : parent(parent), maxSize(batchSize), connID(connID) {
        this->capacity = maxSize*taskSizeHint;
        if (this->capacity){
            this->base_ptr = (char*)malloc(this->capacity+(maxSize*headerSize)+sizeof(int));
            assert(this->base_ptr);
        }
        this->headers_buffer_ptr = (char*)malloc(maxSize*headerSize+sizeof(int));
        assert(this->headers_buffer_ptr);
        this->current_ptr = this->base_ptr;
    }

    virtual int push(message2_t* m){
        this->pushHeader(htonl(m->dest), htonl(m->src), m->type, htobe64(m->size));
        ++size;
        if (m->size){
            size_t distance = current_ptr - base_ptr;
            size_t requiredSpace = m->size;
            if (capacity < distance + requiredSpace){
                this->base_ptr = (char*)realloc(this->base_ptr, distance+2*requiredSpace+maxSize*headerSize+sizeof(int));
                assert(this->base_ptr);
                this->capacity = distance+2*requiredSpace;
                this->current_ptr = this->base_ptr + distance;
            }

            memcpy(this->current_ptr, m->data, m->size); this->current_ptr += m->size;

            if (m->type == ChannelType::FBK) 
                this->flush();
        } else 
            this->flush();
        

        MessageAllocator::releaseMessage(m);

        if (size == maxSize )
            return this->flush();
		return 0;
    }

    ~uBuffer_i(){
        delete headers_buffer_ptr;
    }

    size_t size = 0;
    size_t connID;
};


struct uBuffer_1 : public uBuffer_i {
    uBuffer_1(ff_dsenderMTCL2* parent, size_t connID, size_t batchSize = 1, size_t taskSizeHint = 0) : uBuffer_i(parent, connID, 1, 0) { }
    
    int push(message2_t* m) override {
        size_t totalSize = m->size + headerSize + sizeof(int);

        m->data = (char*)realloc(m->data, totalSize);
        pushHeader_(m->data + m->size, htonl(m->dest), htonl(m->src), m->type, htobe64(m->size));
        *reinterpret_cast<int*>(m->data+totalSize-sizeof(int)) = htonl(1);

        if (this->flush(m->data, totalSize) < 0){
            error("uBuffer_1 sending task (flush Function)");
            return -1;
        }

        MessageAllocator::releaseMessage(m);

        return 0;
    }
};

class ff_dsenderMTCL2: public ff_minode_t<message2_t> { 
    friend class uBuffer_i;
    friend struct uBuffer_1;

    const std::vector<std::tuple<std::string, std::string, std::vector<ff_node*>>>& destEndpoints;
    const std::unordered_map<ff_node*, IngressEgressChannels_t>& channelsDictionary;
protected:
    size_t neos=0;
    size_t totalAcks=0;
    std::unordered_map<int, size_t> dest2ConnID;
    std::vector<MTCL::HandleUser> MTCL_Handlers;
    std::unordered_map<int, std::vector<size_t>> src2PossibleDestConnID;
    std::vector<uBuffer_i*> batchBuffers;
    std::vector<int> handlerCounters;
    int batchSize;
    int messageOTF;
    int coreid;
    std::mt19937 gen{std::random_device{}()};

     int waitAckFrom(int connID){
        while (handlerCounters[connID] == 0){
            for(size_t i = 0; i < handlerCounters.size(); ++i){
                ssize_t r; ack_t a;
                size_t sz;
                if ((r = MTCL_Handlers[i].probe(sz, handlerCounters.size() == 1)) <= 0){ // get blocking only if i need to receive from one handle
                    if (errno == EWOULDBLOCK){
                        assert(r == -1);
                        continue;
                    }
                    perror("recvnnb ack");
                    return -1;
                } else {
                    if (MTCL_Handlers[i].receive(&a, sizeof(ack_t)) == sizeof(ack_t))
                        handlerCounters[i]++;
                    else
                      perror("Receive ack in WaitAckFrom");
                }
                
            }

            // TODO: FIX with better way to pause from receving acks
        }
        return 1;
    }

    uBuffer_i* getMostFilledBuffer(const int& src){
        uBuffer_i* maxBuffer = nullptr;
        auto& possibleDestConnID = src2PossibleDestConnID[src];
        for(auto& connID : possibleDestConnID){
            auto* buffer = batchBuffers[connID];
            if (!maxBuffer) maxBuffer = buffer;
            else if (maxBuffer->size < buffer->size) maxBuffer = buffer;
        }

        if (maxBuffer->size > 0) return maxBuffer;
        size_t connID;
        std::sample(possibleDestConnID.begin(), possibleDestConnID.end(), &connID, 1, gen);
        return batchBuffers[connID];
    }

    
public:

    ff_dsenderMTCL2(const std::vector<std::tuple<std::string, std::string, std::vector<ff_node*>>>& destEndpoints, const std::unordered_map<ff_node*, IngressEgressChannels_t>& channelsDictionary, int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int coreid=-1) : destEndpoints(destEndpoints), channelsDictionary(channelsDictionary), batchSize(batchSize), messageOTF(messageOTF), coreid(coreid) {}

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);

        
        for(auto& [_, endpoint, dest_v] : destEndpoints){
            
            auto h = MTCL::Manager::connect(endpoint, MAX_RETRIES, 10);
            if (!h.isValid()) {
                std::cerr << "Connection failed with this endpoint: " << endpoint << std::endl;
                error("Unable to connect failing!!");
                return -1;
            }

            MTCL_Handlers.push_back(std::move(h));
            handlerCounters.push_back(messageOTF);
            size_t connID = MTCL_Handlers.size() - 1;
            totalAcks += handlerCounters.back();

            if (this->batchSize > 1)
                batchBuffers.push_back(new uBuffer_i(this, connID, this->batchSize));
            else
                batchBuffers.push_back(new uBuffer_1(this, connID));

            for(ff_node* n : dest_v)
                dest2ConnID[n->mioID] = connID;
        }

        // build the map for messages without destination, in order to perform the load balacing
        for(auto& [n, channelsTuple] : channelsDictionary)
            for(auto& [dest_n, t, l, q] : channelsTuple.second)
                if (l == ChannelLocality::REMOTE)
                    src2PossibleDestConnID[n->mioID].push_back(dest2ConnID[dest_n->mioID]);
    

        return 0;
    }

    message2_t *svc(message2_t* task) {
        if (task->dest >= 0)
            batchBuffers[dest2ConnID[task->dest]]->push(task);
        else {

            if (task->isFlush()){
                for (auto& connID_ : std::unordered_set<int>(src2PossibleDestConnID[task->src].begin(), src2PossibleDestConnID[task->src].end()))
                    batchBuffers[connID_]->flush();
                MessageAllocator::releaseMessage(task);
            }
            else if (task->size == 0){
                for (auto& connID_ : std::unordered_set<int>(src2PossibleDestConnID[task->src].begin(), src2PossibleDestConnID[task->src].end()))
                    batchBuffers[connID_]->push(MessageAllocator::make_logical_EOS(task->src));
                
                MessageAllocator::releaseMessage(task);
            } else
                getMostFilledBuffer(task->src)->push(task); // get the most filled buffer socket or a rr socket
        }

        return this->GO_ON;
    }


	void svc_end() {
        for(auto& h : MTCL_Handlers) h.close();

		size_t currentack = 0;
		for(const auto& counter : handlerCounters)	currentack += counter;

		ack_t a;
        ssize_t r;
        size_t sz;
		while(currentack < this->totalAcks) {

            for(size_t i = 0; i < MTCL_Handlers.size(); i++){
                if (handlerCounters[i] == -1) continue;
                if ((r = MTCL_Handlers[i].probe(sz, false)) <= 0){ 
                    if (r==-1 && errno == EWOULDBLOCK){
                        continue;
                    }
                } else {
                    if (MTCL_Handlers[i].receive(&a, sizeof(ack_t)) == sizeof(ack_t)){
                        currentack++;
                        handlerCounters[i]++;
                        continue;
                    }
                }
                currentack += messageOTF - handlerCounters[i];
                handlerCounters[i] = -1;
            }

		}

	}
};



inline int uBuffer_i::flush(){
        // there is nothing to send!
        if (size == 0) return 0;

        memcpy(this->current_ptr, this->headers_buffer_ptr, size*headerSize);
        this->current_ptr += size*headerSize;
        *(reinterpret_cast<int*>(this->current_ptr)) = htonl(this->size);
        this->current_ptr += sizeof(int);

        if (parent->handlerCounters[connID] == 0 && parent->waitAckFrom(connID) == -1){
            error("Errore waiting ack from socket inside the callback\n");
            return -1;
        }

        if (parent->MTCL_Handlers[connID].send(this->base_ptr, this->current_ptr - this->base_ptr ) < 0){
            error("flushing payload buffers");
            return -1;
        }
        
        this->parent->handlerCounters[connID]--;
        this->size = 0;
        this->current_ptr = this->base_ptr;
        return 0;
    }

    inline int uBuffer_i::flush(void* buff, size_t s){

        if (parent->handlerCounters[connID] == 0 && parent->waitAckFrom(connID) == -1){
            error("Errore waiting ack from socket inside the callback\n");
            return -1;
        }

        parent->MTCL_Handlers[connID].send(buff, s);
        
        this->parent->handlerCounters[connID]--;
        return 0;
    }

} // namespace
#endif
