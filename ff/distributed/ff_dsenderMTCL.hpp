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
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_batchbuffer.hpp>
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

class ff_dsenderMTCL;
class ff_dsenderHMTCL;

class uBuffer_i {
    friend class ff_dsenderHMTCL;
protected:
    ff_dsenderMTCL* parent;
    static const size_t headerSize = sizeof(int)+sizeof(int)+sizeof(size_t);
    size_t maxSize, capacity;
    char *base_ptr = nullptr, *current_ptr = nullptr;
    char *headers_buffer_ptr = nullptr;
    int flush(bool nocheck = false);
    

    void pushHeader(int chid, int sender, size_t size_){
        char* offset_ptr = headers_buffer_ptr + this->size*headerSize;
        *reinterpret_cast<int*>(offset_ptr) = sender;
        *reinterpret_cast<int*>(offset_ptr+sizeof(int)) = chid;
        *reinterpret_cast<size_t*>(offset_ptr+2*sizeof(int)) = size_;
    }

public:
    uBuffer_i(ff_dsenderMTCL* parent, size_t connID, size_t batchSize, ChannelType ct, size_t taskSizeHint = 512) : parent(parent), maxSize(batchSize), ct(ct), connID(connID) {
        this->capacity = maxSize*taskSizeHint;
        if (this->capacity){
            this->base_ptr = (char*)malloc(this->capacity);
            assert(this->base_ptr);
        }
        this->headers_buffer_ptr = (char*)malloc(maxSize*headerSize);
        assert(this->headers_buffer_ptr);
        this->current_ptr = this->base_ptr;
    }

    virtual int push(message_t* m){
        this->pushHeader(htonl(m->chid), htonl(m->sender), htobe64(m->data.getLen()));
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
        }

        delete m;

        if (++size == maxSize)
            return this->flush();
		return 0;
    }

    int sendEOS(){
        if (this->push(new message_t(0,0))<0) {
			error("pushing EOS");
		}
        return flush();
    }

    ~uBuffer_i(){
        delete headers_buffer_ptr;
    }

    size_t size = 0;
    ChannelType ct;
    size_t connID;
};


struct uBuffer_1 : public uBuffer_i {
    uBuffer_1(ff_dsenderMTCL* parent, size_t connID, size_t batchSize, ChannelType ct, size_t taskSizeHint = 0) : uBuffer_i(parent, connID, 1, ct, 0) { }
    
    int push(message_t* m) override {
        
        this->pushHeader(htonl(m->chid), htonl(m->sender), htobe64(m->data.getLen()));
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

	
using precomputedRT_t = std::map<std::pair<std::string, ChannelType>, std::vector<int>>;
class ff_dsenderMTCL: public ff_minode_t<message_t> { 
    friend class uBuffer_i;
    friend struct uBuffer_1;
protected:
    size_t neos=0;
    size_t totalAcks=0;
    std::vector<std::pair<ChannelType, ff_endpoint>> dest_endpoints;
    precomputedRT_t* precomputedRT;
    std::map<std::pair<int, ChannelType>, size_t> dest2ConnID;
    std::vector<MTCL::HandleUser> MTCL_Handlers;
    size_t last_rr_connID = 0;
    std::vector<uBuffer_i*> batchBuffers;
    std::vector<int> handlerCounters;
    std::string gName;
    int batchSize;
    int messageOTF, internalmessageOTF;
    int coreid;
 

    virtual int handshakeHandler(MTCL::HandleUser& h, ChannelType t){
        size_t sz = htobe64(gName.size());
        char * buff = new char[sizeof(ChannelType)+sizeof(size_t)+gName.size()];
        memcpy(buff, &t, sizeof(ChannelType));
        memcpy(buff + sizeof(ChannelType), &sz, sizeof(size_t));
        memcpy(buff + sizeof(ChannelType) + sizeof(size_t), gName.c_str(), gName.size());

       if (h.send(buff, sizeof(ChannelType)+sizeof(size_t)+gName.size()) < 0){
            error("Error writing on socket in handshake handler\n");
            return -1;
       }

        return 0;
    }

     int waitAckFrom(int connID){
        while (handlerCounters[connID] == 0){
            for(size_t i = 0; i < handlerCounters.size(); ++i){
            //for(auto& [connID_, counter] : handlerCounters){
                ssize_t r; ack_t a;
                size_t sz;
                if ((r = MTCL_Handlers[i].probe(sz, handlerCounters.size() == 1)) <= 0){ // get blocking only if i need to receive from one handle
                //if ((r = recvnnb(sck_, reinterpret_cast<char*>(&a), sizeof(ack_t))) != sizeof(ack_t)){
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

    size_t getMostFilledBuffer(bool feedback){
        size_t connId_Max = 0;
        size_t sizeMax = 0;
        for(size_t i = 0; i < batchBuffers.size(); i++){
            auto& buffer = batchBuffers[i];
            if ((feedback && buffer->ct != ChannelType::FBK) || (!feedback && buffer->ct != ChannelType::FWD)) continue; 
            if (buffer->size > sizeMax) {
                connId_Max = i;
                sizeMax = buffer->size;
            }
        }
    
        if (sizeMax > 0) return connId_Max;
        
        do {
        last_rr_connID = (last_rr_connID + 1) % this->MTCL_Handlers.size();
        } while (batchBuffers[last_rr_connID]->ct != (feedback ? ChannelType::FBK : ChannelType::FWD));
        return last_rr_connID;
        
    }

    
public:
    ff_dsenderMTCL(std::pair<ChannelType, ff_endpoint> dest_endpoint, precomputedRT_t* rt, std::string gName = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int internalMessageOTF = DEFAULT_INTERNALMSG_OTF, int coreid=-1): precomputedRT(rt), gName(gName), batchSize(batchSize), messageOTF(messageOTF), internalmessageOTF(internalMessageOTF), coreid(coreid) {
        this->dest_endpoints.push_back(std::move(dest_endpoint));
    }

    ff_dsenderMTCL( std::vector<std::pair<ChannelType, ff_endpoint>> dest_endpoints_, precomputedRT_t* rt, std::string gName = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int internalMessageOTF = DEFAULT_INTERNALMSG_OTF, int coreid=-1) : dest_endpoints(std::move(dest_endpoints_)), precomputedRT(rt), gName(gName), batchSize(batchSize), messageOTF(messageOTF), internalmessageOTF(internalMessageOTF), coreid(coreid) {}

    

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
        
        for(auto& [ct, ep] : this->dest_endpoints){
            
            if (ep.address.compare(0, 3, "MPI") == 0){
                ep.address.append(":" + std::to_string(ct+10));
            }

            auto h = MTCL::Manager::connect(ep.address, MAX_RETRIES, 10);
            if (!h.isValid()) {
                std::cerr << "Connection failed with this endpoint: " << ep.address << std::endl;
                error("Unable to connect failing!!");
                return -1;
            }
            MTCL_Handlers.push_back(std::move(h));

            size_t connID = MTCL_Handlers.size() - 1;
            handlerCounters.push_back(ct == ChannelType::INT ? internalmessageOTF : messageOTF);
            totalAcks += handlerCounters.back();

            if (this->batchSize > 1)
                batchBuffers.push_back(new uBuffer_i(this, connID, this->batchSize, ct));
            else
                batchBuffers.push_back(new uBuffer_1(this, connID, 1, ct));

            // compute the routing table!
            for(auto& [k,v] : *precomputedRT){
                if (k.first != ep.groupName || k.second != ct) continue;
                for(int dest : v)
                    dest2ConnID[std::make_pair(dest, ct)] = connID;
            }

            if (handshakeHandler(MTCL_Handlers[connID], ct) < 0) {
				error("svc_init ff_dsender handshake handler failed!");
				return -1;
			}
        }

        // we can erase the list of endpoints
        this->dest_endpoints.clear();

        return 0;
    }

    message_t *svc(message_t* task) {
        size_t connID;
        //if (task->chid == -1) task->chid = 0;
        if (task->chid != -1)
            connID = dest2ConnID[{task->chid, (task->feedback ? ChannelType::FBK : ChannelType::FWD)}];
        else {
            connID = getMostFilledBuffer(task->feedback); // get the most filled buffer socket or a rr socket
        }

        if (batchBuffers[connID]->push(task) == -1) {
			return EOS;
		}

        return this->GO_ON;
    }

    void eosnotify(ssize_t id) {
        // this send the logic EOS signal, not the real one 
        for (size_t i = 0; i < MTCL_Handlers.size(); i++)
            batchBuffers[i]->push(new message_t(id, -2));

		if (++neos >= this->get_num_inchannels()) {
			// all input EOS received, now sending the EOS to all connections
            for(size_t i = 0; i < MTCL_Handlers.size(); i++) {
				if (batchBuffers[i]->sendEOS()<0) {
					error("sending EOS to external connections (ff_dsender)\n");
				}										 
			}
		}
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
                currentack += (batchBuffers[i]->ct == ChannelType::INT ? internalmessageOTF : messageOTF) - handlerCounters[i];
                handlerCounters[i] = -1;
            }

		}

		for(auto& h : MTCL_Handlers) h.close();
	}
};


class ff_dsenderHMTCL : public ff_dsenderMTCL {
    size_t last_rr_connID_Internal = 0;
    bool squareBoxEOS = false;

    size_t getMostFilledInternalBufferConnID(){
         size_t connID_max = 0;
        size_t sizeMax = 0;
        for(auto* buff : batchBuffers){
            if (buff->ct != ChannelType::INT) continue;
            if (buff->size > sizeMax) {
                connID_max = buff->connID;
                sizeMax = buff->size;
            }
        }
        if (sizeMax) return connID_max;

        do {
        last_rr_connID_Internal = (last_rr_connID_Internal + 1) % this->MTCL_Handlers.size();
        } while (batchBuffers[last_rr_connID_Internal]->ct != ChannelType::INT);
        return last_rr_connID_Internal;
    }

public:

    ff_dsenderHMTCL(std::pair<ChannelType, ff_endpoint> e, precomputedRT_t* rt, std::string gName  = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int internalMessageOTF = DEFAULT_INTERNALMSG_OTF, int coreid=-1) : ff_dsenderMTCL(e, rt, gName, batchSize, messageOTF, internalMessageOTF, coreid){} 
    ff_dsenderHMTCL(std::vector<std::pair<ChannelType, ff_endpoint>> dest_endpoints_, precomputedRT_t* rt, std::string gName  = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int internalMessageOTF = DEFAULT_INTERNALMSG_OTF, int coreid=-1) : ff_dsenderMTCL(dest_endpoints_, rt, gName, batchSize, messageOTF, internalMessageOTF, coreid) {}

    message_t *svc(message_t* task) {
        // flush of buffers
            if (task->chid == -2 && task->sender == -2){
                for(auto& bb : batchBuffers)
                    bb->flush();
                delete task;
                return this->GO_ON;
            }
        
        if (this->get_channel_id() == (ssize_t)(this->get_num_inchannels() - 1)){
            size_t connID;

            // pick destination from the list of internal connections!
            if (task->chid != -1){ // roundrobin over the destinations
                connID = dest2ConnID[{task->chid, ChannelType::INT}];
            } else
                connID = getMostFilledInternalBufferConnID();


            if (batchBuffers[connID]->push(task) == -1) {
				return EOS;
			}

            return this->GO_ON;
        }

        return ff_dsenderMTCL::svc(task);
    }

     void eosnotify(ssize_t id) {
         if (id == (ssize_t)(this->get_num_inchannels() - 1)){
            // send the EOS to all the internal connections
            if (squareBoxEOS) return;
            squareBoxEOS = true;
            for(auto* buff: batchBuffers) {
                if (buff->ct != ChannelType::INT) continue;
                if (buff->sendEOS()<0) {
					error("sending EOS to internal connections\n");
				}					
			}
		 }
		 if (++neos >= this->get_num_inchannels()) {
			 // all input EOS received, now sending the EOS to all
			 // others connections
			 for(auto* buff: batchBuffers) {
                 if (buff->ct == ChannelType::INT) continue;
				 if (buff->sendEOS()<0) {
					 error("sending EOS to external connections (ff_dsenderH)\n");
				 }										 
			 }
		 }

        
	 }

	
};

int uBuffer_i::flush(bool nocheck){
        // there is nothing to send!
        if (!nocheck && size == 0) return 0;

        if (this->parent->handlerCounters[connID] == 0 && this->parent->waitAckFrom(connID) == -1){
            error("Errore waiting ack from socket inside the callback\n");
            return false;
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
