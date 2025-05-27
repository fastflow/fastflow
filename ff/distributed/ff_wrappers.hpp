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

#ifndef WRAPPER_H
#define WRAPPER_H

#include <iostream>
#include <type_traits>
#include <functional>
#include <vector>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_ddefines.hpp>
#include <ff/distributed/ff_dutils.hpp>
#include <ff/distributed/ff_messageAllocator.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>

namespace ff {

class WrapperINOUT: public internal_mi_transformer {

private:
    const IngressEgressChannels_t& channelsInfo;
	std::unordered_map<int, inputMapRecord_t> inputMap;
	std::vector<outputMapRecord_t> outMap;
	bool sink;
	size_t eos_received = 0;

	message2_t* reusableMsg = nullptr;
public:

	WrapperINOUT(ff_node* n, const IngressEgressChannels_t& channels, bool cleanup=false) : internal_mi_transformer(this, cleanup), channelsInfo(channels) {
		this->n = n;
		this->mioID = n->mioID;
		// registering ff_send_out callback
		internal_mi_transformer::registerCallback(ff_send_out_to_cbk, this);
	}

    bool serialize(void* in, int id, bool ret = false) {
		if (sink) return true;
		if (in == EOS){
			ff_minode::ff_send_out(MessageAllocator::make_logical_EOS(this->n->mioID, id));
			if (ret) ff_minode::ff_send_out(in);
			return true;
		}

		if (in == FLUSH){
			ff_minode::ff_send_out(MessageAllocator::make_flush(this->n->mioID));
			return true;
		}

		if (ret && in > FF_TAG_MIN) return ff_minode::ff_send_out(in);
		
		// try to reuse an already present message
		message2_t* msg = reusableMsg ? reusableMsg : MessageAllocator::allocateMessage();

		msg->src = this->n->mioID;
		if (id == -1){
			msg->dest = -1;
			msg->type = ChannelType::FWD;
		} else {
			auto& chInfo = outMap[id];
			msg->dest = chInfo.identifier;
			msg->type = chInfo.type;
		}

		msg->locality = ChannelLocality::REMOTE;

		bool datacopied = this->n->serializeF(in, msg);
		msg->cleanup = true;
		msg->freeCallback = this->n->freeBlob;
	
		ff_minode::ff_send_out(msg);
		if (datacopied)
			this->n->freetaskF(in);
			
		return true;
	}

	int svc_init() {
		sink = channelsInfo.second.size() == 0;
		if (this->n->isMultiInput())
			reinterpret_cast<ff_minode*>(this->n)->set_running(channelsInfo.first.size());
		
		if (this->n->isMultiOutput()){
			ff_monode* monode = reinterpret_cast<ff_monode*>(this->n);
			monode->set_virtual_outchannels(std::count_if(channelsInfo.second.begin(), channelsInfo.second.end(), [](auto& o){return std::get<1>(o) == ChannelType::FWD;}));
			monode->set_virtual_feedbackchannels(std::count_if(channelsInfo.second.begin(), channelsInfo.second.end(), [](auto& o){return std::get<1>(o) == ChannelType::FBK;}));
		}

		// build the input map
		for(size_t i = 0; i < channelsInfo.first.size(); i++){
			auto& t = channelsInfo.first[i];
			inputMap[std::get<0>(t)->mioID] = inputMapRecord_t(i, std::get<1>(t), std::get<2>(t));
		}

		if (!std::count_if(channelsInfo.first.begin(), channelsInfo.first.end(), [](auto& o){return std::get<1>(o) == ChannelType::FWD;}))
			this->skipfirstpop(true);
			

		// build the output map
		for(size_t i = 0; i < channelsInfo.second.size(); i++){
			auto& t = channelsInfo.second[i];
			outMap.emplace_back(std::get<0>(t)->mioID, std::get<1>(t), std::get<2>(t));
		}

		// set the size of output queue to sender to 1
		size_t oldsz;
        ff::svector<ff_node*> realOutputsNodes; this->get_out_nodes(realOutputsNodes);
        realOutputsNodes.back()->change_outputqueuesize(1, oldsz);

		return n->svc_init();
	}

	void eosnotify(ssize_t id){	}

	void svc_end(){this->n->svc_end();}
	
	void * svc(void* in) {
		void* out;
		if (in != nullptr) {
			message2_t* msg = (message2_t*)in;
			
			if (msg->size == 0){ // logical EOS!
				inputMapRecord_t& record = inputMap[msg->src];
				if (!record.eos_received){
					this->n->eosnotify(record.index);
					record.eos_received = true;

					if (++eos_received >= this->channelsInfo.first.size()){
						if (!sink) // if i'm not a sink propagate the logical EOS
							ff_minode::ff_send_out(MessageAllocator::make_logical_EOS(this->n->mioID));
						
						// de-registering the call-back to allow the EOS to flow without being captured by the callback
						internal_mi_transformer::registerCallback([](void* task, int id, unsigned long retry, unsigned long ticks, void * obj) {
							return reinterpret_cast<WrapperINOUT*>(obj)->ff_minode::ff_send_out(task, id, retry, ticks);
						}, this);

						delete msg;
						return EOS;
					}
				}
				delete msg;
				return GO_ON;
			}
			
			if (this->n->isMultiInput())
				reinterpret_cast<ff_minode*>(this->n)->set_input_channelid(inputMap[msg->src].index, msg->type == ChannelType::FWD);

			bool datacopied=true;
			out = n->svc(this->n->deserializeF(msg, datacopied, n));
			if (!datacopied) msg->cleanup = false;
			
			// try to reuse the message for an eventual output
			if (reusableMsg) 
				MessageAllocator::releaseMessage(msg);
			else {
				reusableMsg = msg;
				reusableMsg->cleanContent();
			}

		}  else // it can happen if we have a feedback channel
			out = n->svc(nullptr);
		
        serialize(out, -1, true);
        return GO_ON;
	}

	~WrapperINOUT(){
		if (reusableMsg) delete reusableMsg;
	}

	bool init_output_blocking(pthread_mutex_t   *&m,
							  pthread_cond_t    *&c,
							  bool feedback=true) {
        return ff_node::init_output_blocking(m,c,feedback);
    }

    void set_output_blocking(pthread_mutex_t   *&m,
							 pthread_cond_t    *&c,
							 bool canoverwrite=false) {
		ff_node::set_output_blocking(m,c,canoverwrite);
	}

    static inline bool ff_send_out_to_cbk(void* task, int id,
										  unsigned long retry,
										  unsigned long ticks, void* obj) {		
		return ((WrapperINOUT*)obj)->serialize(task, id);
	}
};

} // namespace
#endif
