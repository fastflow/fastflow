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
//class ff_dsenderHMTCL;

class uBuffer_i {
    friend class ff_dsenderHMTCL;
protected:
    ff_dsenderMTCL2* parent;
    static const size_t headerSize = sizeof(int)+sizeof(int)+sizeof(ChannelType)+sizeof(size_t);
    size_t maxSize, capacity;
    char *base_ptr = nullptr, *current_ptr = nullptr;
    char *headers_buffer_ptr = nullptr;
    int flush(bool nocheck = false);
    

    void pushHeader(int dest, int src, ChannelType t, size_t size_){
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
            this->base_ptr = (char*)malloc(this->capacity);
            assert(this->base_ptr);
        }
        this->headers_buffer_ptr = (char*)malloc(maxSize*headerSize);
        assert(this->headers_buffer_ptr);
        this->current_ptr = this->base_ptr;
    }

    virtual int push(message2_t* m){
        this->pushHeader(htonl(m->dest), htonl(m->src), m->type, htobe64(m->data.getLen()));
        ++size;
        if (m->data.getLen()){
            size_t distance = current_ptr - base_ptr;
            size_t requiredSpace = m->data.getLen();
            if (capacity < distance + requiredSpace){
                this->base_ptr = (char*)realloc(this->base_ptr, distance+2*requiredSpace);
                assert(this->base_ptr);
                this->capacity = distance+2*requiredSpace;
                this->current_ptr = this->base_ptr + distance;
            }

            memcpy(this->current_ptr, m->data.getPtr(), m->data.getLen()); this->current_ptr += m->data.getLen();
        } else 
            this->flush();
        

        delete m;

        if (size == maxSize )
            return this->flush();
		return 0;
    }

    int sendEOS(){
        if (this->push(message2_t::make_pyshical_EOS())<0) {
			error("pushing EOS");
		}
        return flush();
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
        
        this->pushHeader(htonl(m->dest), htonl(m->src), m->type, htobe64(m->data.getLen()));
        if (m->data.getLen()){
            this->base_ptr = m->data.getPtr();
            this->current_ptr = this->base_ptr + m->data.getLen();
        }
        this->size = 1;
        if (this->flush(true) < 0){
            error("uBuffer_1 sending task (flush Function)");
            return -1;
        }

        delete m;

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
 

    virtual int handshakeHandler(MTCL::HandleUser& h){
        /*size_t sz = htobe64(gName.size());
        char * buff = new char[sizeof(ChannelType)+sizeof(size_t)+gName.size()];
        memcpy(buff, &t, sizeof(ChannelType));
        memcpy(buff + sizeof(ChannelType), &sz, sizeof(size_t));
        memcpy(buff + sizeof(ChannelType) + sizeof(size_t), gName.c_str(), gName.size());

       if (h.send(buff, sizeof(ChannelType)+sizeof(size_t)+gName.size()) < 0){
            error("Error writing on socket in handshake handler\n");
            return -1;
       }*/

        return 0;
    }

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
            
            if (handshakeHandler(MTCL_Handlers[connID]) < 0) {
				error("svc_init ff_dsender handshake handler failed!");
				return -1;
			}
        }

        // build the map for messages without destination, in order to perform the load balacing
        for(auto& [n, channelsTuple] : channelsDictionary)
            for(auto& [dest_n, t, l, q] : channelsTuple.second)
                if (l == ChannelLocality::REMOTE)
                    src2PossibleDestConnID[n->mioID].push_back(dest2ConnID[dest_n->mioID]);
    

        return 0;
    }

    message2_t *svc(message2_t* task) {
        if (task->dest != -1)
            batchBuffers[dest2ConnID[task->dest]]->push(task);
        else {
            if (task->data.getLen() == 0){
                for (auto& connID_ : std::unordered_set<int>(src2PossibleDestConnID[task->src].begin(), src2PossibleDestConnID[task->src].end())){
                    batchBuffers[connID_]->push(message2_t::make_logical_EOS(task->src));
                }
                delete task;
            } else
                getMostFilledBuffer(task->src)->push(task); // get the most filled buffer socket or a rr socket
        }

        return this->GO_ON;
    }

    void eosnotify(ssize_t id) {
        // this send the logic EOS signal, not the real one 
        //for (size_t i = 0; i < MTCL_Handlers.size(); i++)
        //    batchBuffers[i]->push(new message2_t); //TODO: fix 

		/*if (++neos >= this->get_num_inchannels()) {
			// all input EOS received, now sending the EOS to all connections
            for(size_t i = 0; i < MTCL_Handlers.size(); i++) {
				if (batchBuffers[i]->sendEOS()<0) {
					error("sending EOS to external connections (ff_dsender)\n");
				}										 
			}
		}*/


    }

	void svc_end() {
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

		for(auto& h : MTCL_Handlers) h.close();
	}
};



int uBuffer_i::flush(bool nocheck){
        // there is nothing to send!
        if (!nocheck && size == 0) return 0;

        if (parent->handlerCounters[connID] == 0 && parent->waitAckFrom(connID) == -1){
            error("Errore waiting ack from socket inside the callback\n");
            return -1;
        }
        MTCL::Request h_req;
        if (parent->MTCL_Handlers[connID].isend(this->headers_buffer_ptr, size*headerSize, h_req) < 0){
            error("Flushing header buffers");
            return -1;
        }
        MTCL::Request p_req;
        if (this->current_ptr != this->base_ptr)
            if (parent->MTCL_Handlers[connID].isend(this->base_ptr, this->current_ptr - this->base_ptr, p_req) < 0){
                error("flushing payload buffers");
                return -1;
            }
        
        MTCL::waitAll(h_req, p_req);
        
        this->parent->handlerCounters[connID]--;
        this->size = 0;
        this->current_ptr = this->base_ptr;
        return 0;
    }

} // namespace
#endif
