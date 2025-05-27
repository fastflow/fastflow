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
 *
 * Authors: 
 *   Nicolo' Tonci
 *   Massimo Torquati
 */

#ifndef ADAPTERS_H
#define ADAPTERS_H

#include <iostream>
#include <type_traits>
#include <functional>
#include <random>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_ddefines.hpp>
#include <ff/distributed/ff_dutils.hpp>
#include <ff/distributed/ff_messageAllocator.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>


namespace ff {

template <typename Iterator, typename T>
int getIndex(const Iterator& start_it, const Iterator& stop_it, const T& v){
    auto it = std::find(start_it, stop_it, v);
    if (it == stop_it) return -1;
    return std::distance(start_it, it);
}

class EmitterAdapter2: public internal_mo_transformer {
private:
    int nextDestination = -1;
	const EgressChannels_t& channelsInfo;
	const std::vector<ff_node*>& nextLocalNodes;
	std::vector<outputMapRecord_t> outMap;
	std::vector<int> localDest;
	bool exited = false;
	ChannelType defaultChannelType = ChannelType::FWD;
public:
	EmitterAdapter2(ff_node* n, const EgressChannels_t& channels, const std::vector<ff_node*>& nextLocalNodes, bool skipallpop = false, bool cleanup=false): internal_mo_transformer(this, cleanup), channelsInfo(channels), nextLocalNodes(nextLocalNodes) {
		if (skipallpop) this->skipallpop(true);
		this->n       = n;
		this->mioID = n->mioID;
		internal_mo_transformer::registerCallback(ff_send_out_to_cbk, this);
	}

	int svc_init() {
		size_t numFeedbackChannels = std::count_if(channelsInfo.begin(), channelsInfo.end(), [](auto& o){return std::get<1>(o) == ChannelType::FBK;});
		if (numFeedbackChannels) defaultChannelType = ChannelType::FBK;
		if (this->n->isMultiOutput()) {
			ff_monode* monode = reinterpret_cast<ff_monode*>(this->n);
			monode->set_virtual_outchannels(std::count_if(channelsInfo.begin(), channelsInfo.end(), [](auto& o){return std::get<1>(o) == ChannelType::FWD;}));
			monode->set_virtual_feedbackchannels(numFeedbackChannels);
		}

		// expand the building block in the next level of the group
		ff::svector<int> expandedNextLocalWorkers;
		for(auto* bb: nextLocalNodes)
			custom_get_in_nodes(bb, expandedNextLocalWorkers);

		ff::svector<ff_node*> pyhsicalNextLocalWorkers; this->get_out_nodes(pyhsicalNextLocalWorkers);
		size_t oldsz;
		// build the output map
		for(size_t i = 0; i < channelsInfo.size(); i++){
			auto& t = channelsInfo[i];
			auto indexOfLocalPlacement = getIndex(expandedNextLocalWorkers.begin(), expandedNextLocalWorkers.end(), std::get<0>(t)->mioID);
			if (indexOfLocalPlacement != -1) {
				localDest.push_back(indexOfLocalPlacement);
				if (std::get<3>(t) > 0) {
					assert(indexOfLocalPlacement < (ssize_t)this->get_num_outchannels()-1); // check that we are not going to change the queue size of the squarebox
					pyhsicalNextLocalWorkers[indexOfLocalPlacement]->change_inputqueuesize(std::get<3>(t), oldsz);
				}
			}
			outMap.emplace_back(std::get<0>(t)->mioID, std::get<1>(t), std::get<2>(t), indexOfLocalPlacement);
		}
		
		pyhsicalNextLocalWorkers.back()->change_inputqueuesize(1, oldsz);
		

		return n->svc_init();
	}
	
	void * svc(void* in) {
		void* out = n->svc(in);
						
        if (!this->forward(out, -1, true)) return out;
		
		return GO_ON;
	}

	void eosnotify(ssize_t){
		if (!this->exited){
			this->n->eosnotify();
			ff_monode::ff_send_out_to(MessageAllocator::make_logical_EOS(this->n->mioID), this->get_num_outchannels()-1);

			for(auto& d : localDest)
				ff_monode::ff_send_out_to(this->LEOS, d);
			this->exited = true;
		}
	}

    bool forward(void* task, int destination, bool ret = false){
		if (task == EOS){
			ff_monode::ff_send_out_to(MessageAllocator::make_logical_EOS(this->n->mioID), this->get_num_outchannels()-1);
			for(auto& d : localDest)
				ff_monode::ff_send_out_to(this->LEOS, d);
			this->exited =true;
			if (ret) return false;
			return true;
		}

		if (task == FLUSH){
			ff_monode::ff_send_out_to(MessageAllocator::make_flush(this->n->mioID), this->get_num_outchannels()-1);
			return true;
		}
		
		if (task > FF_TAG_MIN)
			return false;

		if (destination == -1){

			message2_t* msg = nullptr;
			bool datacopied = true;
			do {
				for(size_t i = 0; i < localDest.size(); i++){
					const int idx = (nextDestination + 1 + i) % localDest.size();
					if (ff_send_out_to(task, localDest[idx], 1)){ // non blcking ff_send_out_to, we try just once                                                         
						nextDestination = idx;
                        if (msg) {
							if (!datacopied) msg->cleanup = false;
							MessageAllocator::releaseMessage(msg);
						}
                        return true;
					}
				}
				if (!msg) {
					msg = MessageAllocator::allocateMessage();
					msg->src = this->n->mioID;
					msg->dest = -1;
					msg->locality = ChannelLocality::REMOTE; // correct
					msg->type = defaultChannelType;
					msg->cleanup = true;
					datacopied = this->n->serializeF(task, msg);
					if (!datacopied) {
						msg->freeCallback = this->n->freeBlob;
					}
				}
				if (ff_send_out_to(msg, this->get_num_outchannels() - 1, 1)) {
					if (datacopied) this->n->freetaskF(task);
					return true;
				}
			} while(1);

		} else {
			
			auto& chInfo = outMap[destination];
			if (chInfo.localIndex != -1){
				ff_send_out_to(task, chInfo.localIndex);
				return true;
			}
			message2_t* msg = MessageAllocator::allocateMessage();
			msg->src = this->n->mioID;
			msg->dest = chInfo.identifier;
			msg->type = chInfo.type;
			msg->locality = chInfo.locality;

			bool datacopied = this->n->serializeF(task, msg);
			msg->cleanup = true;
			msg->freeCallback = this->n->freeBlob;
			ff_send_out_to(msg, this->get_num_outchannels() - 1); // send to the routing worker
			if (datacopied) this->n->freetaskF(task);
			return true;
		}
    }

	void svc_end(){n->svc_end();}
	
	int run(bool skip_init=false) {
		return internal_mo_transformer::run(skip_init);
	}
	
	static inline bool ff_send_out_to_cbk(void* task, int id,
										  unsigned long retry,
										  unsigned long ticks, void* obj) {
        return ((EmitterAdapter2*)obj)->forward(task, id);
	}
};

class CollectorAdapter2: public internal_mi_transformer {

private:
	const IngressChannels_t& channelsInfo;
	const std::vector<ff_node*>&  prevLocalNodes;
	std::unordered_map<int, inputMapRecord_t> inputMap;
	std::vector<std::pair<int, ChannelType>> localInputMap;
	size_t eos_received = 0;
	bool prevLevelSquareBox;
public:
	CollectorAdapter2(ff_node* n, const IngressChannels_t& channels, const std::vector<ff_node*>& prevLocalNodes, bool prevLevelSquareBox = true, bool cleanup=false): internal_mi_transformer(this, cleanup), channelsInfo(channels), prevLocalNodes(prevLocalNodes), prevLevelSquareBox(prevLevelSquareBox) {
		this->n       = n;
		this->mioID = n->mioID;
	}

	int svc_init() {
		if (this->n->isMultiInput())
			reinterpret_cast<ff_minode*>(this->n)->set_virtual_inchannels(channelsInfo.size());
		
		ff::svector<int> expandedPrevLocalNodes;
		for(auto* bb : prevLocalNodes)
			custom_get_out_nodes(bb, expandedPrevLocalNodes);

		localInputMap = std::vector<std::pair<int, ChannelType>>(expandedPrevLocalNodes.size(), {-1, ChannelType::FWD});

		// if i have just feedback channels skip just the first pop
		if (std::count_if(channelsInfo.begin(), channelsInfo.end(), [](auto& t){return std::get<1>(t) == ChannelType::FWD;}) == 0)
			this->skipfirstpop(true);

		for(size_t i = 0; i < channelsInfo.size(); i++){
			auto& t = channelsInfo[i];
			auto idx = getIndex(expandedPrevLocalNodes.begin(), expandedPrevLocalNodes.end(), std::get<0>(t)->mioID);
			if (idx == -1)
				inputMap[std::get<0>(t)->mioID] = inputMapRecord_t(i, std::get<ChannelType>(t), std::get<ChannelLocality>(t));
			else
				localInputMap[idx] = {i, std::get<ChannelType>(t)};
		}

		return this->n->svc_init();
	}

	void eosnotify(ssize_t){
		// catch the real EOS and discard
	}

	void svc_end(){this->n->svc_end();}
	
	void * svc(void* in) {
		if (in != nullptr){
			ssize_t channel;
			ChannelType type = ChannelType::FWD;
			// if the results come from the "square box" or "receiver", it is a result from a remote workers so i have to read from which worker it come from 
			if (prevLevelSquareBox && this->fromInput() && (size_t)get_channel_id() == (this->get_num_inchannels() - this->get_num_feedbackchannels() - 1)){
				message2_t * msg = reinterpret_cast<message2_t*>(in);
				if (inputMap.count(msg->src) == 0) {
					std::cerr << "Errore ho ricevuto quLCOSA CHE NON DOVEVO\n";
					abort();
				}
				auto& channelInfo = inputMap[msg->src];
				
				if (msg->size == 0){ // EOS logico
					if (!channelInfo.eos_received){
						this->n->eosnotify(channelInfo.index); 
						channelInfo.eos_received = true;
						if (++eos_received >= this->channelsInfo.size()){
							MessageAllocator::releaseMessage(msg);
							return EOS;
						}
					}
					MessageAllocator::releaseMessage(msg);
					return GO_ON;
				}

				channel = channelInfo.index;
				type = channelInfo.type; 
				bool datacopied = true;
				in = this->n->deserializeF(msg, datacopied, this->n);
				if (!datacopied) msg->cleanup = false;
				MessageAllocator::releaseMessage(msg);
			} else {  // the result come from a local worker, just pass it to collector and compute the right worker id
				auto& channelInfo = localInputMap[get_channel_id()];
				channel = channelInfo.first;
				type = channelInfo.second;
				if (in == this->LEOS) {
					this->n->eosnotify(channel); // TODO: msg->sender here is not consistent... always 0
					if (++eos_received >= this->channelsInfo.size()){
						return EOS;
					}
					return GO_ON;
				}
			}

			// update the input channel id field only if the wrapped node is a multi input
			if (this->n->isMultiInput()) 
				reinterpret_cast<ff_minode*>(this->n)->set_input_channelid(channel, type == ChannelType::FWD);
		}

		return n->svc(in);
	}
};

class SquareBox : public ff_monode {
	const std::vector<ff_node*>& nextLocalNodes;
	const std::unordered_map<ff_node*, IngressEgressChannels_t>& channelsDictionary;
	std::unordered_map<int, std::vector<int>> localLBMap; // local map for load balacing of messages without destinations
    int rr_dest;
#ifdef RANDOM_LOADBALANCING
	std::mt19937 gen{std::random_device{}()};
#endif
	bool nextLevelSquareBox;
public:
	/*
	 *  - localWorkers: list of pairs <logical_destination, physical_destination> where logical_destination is the original destination of the shared-memory graph
	 */
	SquareBox(const std::vector<ff_node*>& nextLocalNodes, const std::unordered_map<ff_node*, IngressEgressChannels_t>& channelsDictionary, bool nextLevelSquareBox = true) : nextLocalNodes(nextLocalNodes), channelsDictionary(channelsDictionary), nextLevelSquareBox(nextLevelSquareBox) {}
	std::unordered_map<int, int> localLookup;

	int svc_init(){
		
		ff::svector<int> expandedNextLocalWorkers;
		for(auto* bb: nextLocalNodes)
			custom_get_in_nodes(bb, expandedNextLocalWorkers); // expand the next local workers and put the ID in the svector
		
		ff::svector<ff_node*> realOutputsNodes; this->get_out_nodes(realOutputsNodes);
		size_t oldsz;

		for(size_t i = 0; i < expandedNextLocalWorkers.size(); i++){
			localLookup[expandedNextLocalWorkers[i]] = i;
			
			auto it = std::find_if(channelsDictionary.begin(), channelsDictionary.end(), [&](auto& k){return k.first->mioID == expandedNextLocalWorkers[i];});
			if (it != channelsDictionary.end()){
				for(auto& t : it->second.first)
					localLBMap[std::get<0>(t)->mioID].push_back(i);
				
				// if the worker is part of an ondemand bb, so the queue length was different from the default one we should set that legth accordingly
				if (!it->second.first.empty()){
					auto& ci = it->second.first.front();
					int newsize = std::get<3>(ci);
					if (newsize > 0)
						realOutputsNodes[i]->change_inputqueuesize(newsize, oldsz);
				}
			}

		}

		if (nextLevelSquareBox) // if there is the square box in the next level set the queue size to 1
			realOutputsNodes.back()->change_inputqueuesize(1, oldsz);

		return 0;
	}
    
	void* svc(void* in){
		message2_t* msg = reinterpret_cast<message2_t*>(in);
		if (msg->isFlush())
			ff_send_out_to(msg, this->get_num_outchannels()-1);
		else if (msg->dest == -1) {
			// load balacing without fixed destination
			if (localLBMap.count(msg->src) == 0) {
				if (nextLevelSquareBox) ff_send_out_to(msg, this->get_num_outchannels()-1);
				else {
					std::cerr << "LOST MESSAGE!!!!!\n";
					MessageAllocator::releaseMessage(msg);
				}
				return this->GO_ON;
			}
			
			if (msg->size == 0){ // this is a logical EOS, without destination we should broadcast it 
				std::vector<int>& possibleDest = localLBMap[msg->src];

				for(auto& dest : possibleDest)
					ff_send_out_to(MessageAllocator::make_logical_EOS(msg->src, -1), dest);
				
				MessageAllocator::releaseMessage(msg);
				return this->GO_ON;
			}

			if (!nextLevelSquareBox) // if there is no square box, it means all the next works can receive the task (this is not always true so this line can introduce some bugs)
				ff_send_out(in);
			else {
				int dest;
				std::vector<int>& possibleDest = localLBMap[msg->src];
				do {
#ifdef RANDOM_LOADBALANCING
					std::sample(possibleDest.begin(), possibleDest.end(), &dest, 1, gen);
#else
					rr_dest = (rr_dest + 1) % possibleDest.size();
					dest = possibleDest[rr_dest];
#endif
				} while (!ff_send_out_to(in, dest, 1));
			}
		}
		else if (localLookup.count(msg->dest)) 
			ff_send_out_to(in, localLookup[msg->dest]);
		else
			ff_send_out_to(in, this->get_num_outchannels()-1); 		// send to the next square box or to the sender (it should work most of the cases)
		
		
		return this->GO_ON;
    }
};

struct SquareBoxInputAdapter : public ff_minode {

	void* svc(void* in) {return in;}

	void eosnotify(ssize_t id){
		if (this->fromInput() && (id == (ssize_t)(this->get_num_inchannels() - this->get_num_feedbackchannels() - 1)))
			ff_send_out(this->EOS);
	}
};

} // namespace
#endif
