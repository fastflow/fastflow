#ifndef ADAPTERS_H
#define ADAPTERS_H

#include <iostream>
#include <type_traits>
#include <functional>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>

#define FARM_GATEWAY -10

using namespace ff;
template<typename T>
struct TaskWrapper {
    T* task;
    int destination;
    TaskWrapper(T* task, int d) : task(task), destination(d) {}
};

template<typename T>
struct ResultWrapper {
    T* result;
    int source;
    ResultWrapper(T* result, int s): result(result), source(s) {}
};

template<typename T>
class SquareBoxEmitter : public ff_minode_t<TaskWrapper<T>, SMmessage_t> {
    std::vector<int> sources;
	int neos = 0;
public:
	SquareBoxEmitter(const std::vector<int> localSources) : sources(localSources) {}
	
	SMmessage_t* svc(TaskWrapper<T>* in) {
			size_t chId = ff_minode::get_channel_id();
			SMmessage_t* o = new SMmessage_t(in->task, chId < sources.size() ? sources[chId] : -1 , -100 - in->destination);
			delete in;
            return o;
        }

	void eosnotify(ssize_t id) {
		if (id == (ssize_t)sources.size()) return;   // EOS coming from the SquareBoxCollector, we must ignore it
		if (++neos == (int)sources.size()){
			this->ff_send_out(this->EOS);
		}
	}
};

template<typename T>
class SquareBoxCollector : public ff_monode_t<SMmessage_t, ResultWrapper<T>> {
	std::unordered_map<int, int> destinations;
public:
	SquareBoxCollector(const std::unordered_map<int, int> localDestinations) : destinations(localDestinations) {}
    
	ResultWrapper<T>* svc(SMmessage_t* in){
		this->ff_send_out_to(new ResultWrapper<T>(reinterpret_cast<T*>(in->task), in->sender), destinations[in->dst]);
		delete in; return this->GO_ON;
    }
};

template<typename Tin, typename Tout = Tin>
class EmitterAdapter: public internal_mo_transformer {
private:
    int totalWorkers;
	std::unordered_map<int, int> localWorkersMap;
    int nextDestination;
public:

	typedef Tout T_out;
	
	EmitterAdapter(ff_node_t<Tin, Tout>* n, int totalWorkers, std::unordered_map<int, int> localWorkers = {0,0}, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers), localWorkersMap(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}

	EmitterAdapter(ff_node* n, int totalWorkers, std::unordered_map<int, int> localWorkers = {0,0}, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers),  localWorkersMap(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}
	
	EmitterAdapter(ff_monode_t<Tin, Tout>* n, int totalWorkers, std::unordered_map<int, int> localWorkers = {0,0}, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers), localWorkersMap(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}

	EmitterAdapter(ff_minode_t<Tin, Tout>* n, int totalWorkers, std::unordered_map<int, int> localWorkers = {0,0}, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers), localWorkersMap(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}

	template <typename T>
	EmitterAdapter(ff_comb_t<Tin, T, Tout>* n, int totalWorkers, std::unordered_map<int, int> localWorkers = {0,0}, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers), localWorkersMap(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}

	void * svc(void* in) {
		void* out = n->svc(in);
		if (out > FF_TAG_MIN) return out;					
		
        this->forward(out, -1);

		return GO_ON;
	}

    bool forward(void* task, int destination){
        if (destination == -1) {
			destination = nextDestination;
			nextDestination = (nextDestination + 1) % totalWorkers;
		}

		
        
        auto pyshicalDestination = localWorkersMap.find(destination);
		if (pyshicalDestination != localWorkersMap.end()) {
			ff_send_out_to(task, pyshicalDestination->second);
		} else {
			ff_send_out_to(new TaskWrapper<Tout>(reinterpret_cast<Tout*>(task), destination), localWorkersMap.size());
		}
        
        return true;
    }

	int svc_init() {
		if (this->n->isMultiOutput()) {
			ff_monode* mo = reinterpret_cast<ff_monode*>(this->n);
			mo->set_running(localWorkersMap.size() + 1); // the last worker is the forwarder to the remote workers
		}
		return n->svc_init();
	}

	void svc_end(){n->svc_end();}
	
	int run(bool skip_init=false) {
		return internal_mo_transformer::run(skip_init);
	}
	
	ff::ff_node* getOriginal(){return this->n;}
	
	static inline bool ff_send_out_to_cbk(void* task, int id,
										  unsigned long retry,
										  unsigned long ticks, void* obj) {
        return ((EmitterAdapter*)obj)->forward(task, id);
	}
};


template<typename Tin, typename Tout = Tin>
class CollectorAdapter: public internal_mi_transformer {

private:
	std::vector<int> localWorkers;
public:

	CollectorAdapter(ff_node_t<Tin, Tout>* n, std::vector<int> localWorkers, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	CollectorAdapter(ff_node* n, std::vector<int> localWorkers, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	CollectorAdapter(ff_minode_t<Tin, Tout>* n, std::vector<int> localWorkers, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	CollectorAdapter(ff_monode_t<Tin, Tout>* n, std::vector<int> localWorkers, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	template <typename T>
	CollectorAdapter(ff_comb_t<Tin, T, Tout>* n, std::vector<int> localWorkers, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
		internal_mi_transformer::registerCallback(cb,arg);
	}

	int svc_init() {
		if (this->n->isMultiInput()) { ////???????? MASSIMO???
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_running(localWorkers.size() + 1);
		}
		return n->svc_init();
	}

	void svc_end(){this->n->svc_end();}
	
	void * svc(void* in) {
        Tin * task;
        ssize_t channel;

		// if the results come from the "square box", it is a result from a remote workers so i have to read from which worker it come from 
		if ((size_t)get_channel_id() == localWorkers.size()){
			ResultWrapper<Tin> * tw = reinterpret_cast<ResultWrapper<Tin>*>(in);
            task = tw->result;
            channel = tw->source;
			delete tw;
		} else {  // the result come from a local worker, just pass it to collector and compute the right worker id
			task = reinterpret_cast<Tin*>(in);
            channel = localWorkers.at(get_channel_id());
		}

		// update the input channel id field only if the wrapped node is a multi input
		if (this->n->isMultiInput()) {
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_input_channelid(channel, true);
		}	

		return n->svc(task);
	}

	ff_node* getOriginal(){ return this->n;	}
};

#endif
