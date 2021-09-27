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
class SquareBoxEmitter : public ff_monode_t<TaskWrapper<T>, T> {
    T* svc(TaskWrapper<T>* in) {
            this->ff_send_out_to(in->task, in->destination);
            return this->GO_ON;
        }
};

template<typename T>
class SquareBoxCollector : public ff_minode_t<T, ResultWrapper<T>> {
    ResultWrapper<T>* svc(T* in){
        return new ResultWrapper<T>(in, ff_minode::get_channel_id());
    }
};


template<typename Tin, typename Tout = Tin>
class EmitterAdapter: public internal_mo_transformer {
private:
    int totalWorkers, localWorkers;
    int nextDestination;
public:

	typedef Tout T_out;
	
	EmitterAdapter(ff_node_t<Tin, Tout>* n, int totalWorkers, int localWorkers=1, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers), localWorkers(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}

	EmitterAdapter(ff_node* n, int totalWorkers, int localWorkers=1, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers),  localWorkers(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}
	
	EmitterAdapter(ff_monode_t<Tin, Tout>* n, int totalWorkers, int localWorkers=1, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers), localWorkers(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}

	EmitterAdapter(ff_minode_t<Tin, Tout>* n, int totalWorkers, int localWorkers=1, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers), localWorkers(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}

	template <typename T>
	EmitterAdapter(ff_comb_t<Tin, T, Tout>* n, int totalWorkers, int localWorkers=1, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers), localWorkers(localWorkers) {
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
        if (destination == -1) destination = nextDestination;
        
        if (destination < localWorkers) ff_send_out_to(task, destination);
            else ff_send_out_to(new TaskWrapper<Tout>(reinterpret_cast<Tout*>(task), destination), localWorkers);

        nextDestination = (nextDestination + 1) % totalWorkers;
        return true;
    }

	int svc_init() {
		if (this->n->isMultiOutput()) {
			ff_monode* mo = reinterpret_cast<ff_monode*>(this->n);
			mo->set_running(localWorkers + 1); // the last worker is the forwarder to the remote workers
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
    int localWorkers, indexFirstLocal;	// number of input channels the wrapped node is supposed to have
public:

	CollectorAdapter(ff_node_t<Tin, Tout>* n, int indexFirstLocal, int localWorkers=1, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers), indexFirstLocal(indexFirstLocal) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	CollectorAdapter(ff_node* n, int indexFirstLocal, int localWorkers=1, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers), indexFirstLocal(indexFirstLocal) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	CollectorAdapter(ff_minode_t<Tin, Tout>* n, int indexFirstLocal, int localWorkers, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers), indexFirstLocal(indexFirstLocal) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	CollectorAdapter(ff_monode_t<Tin, Tout>* n, int indexFirstLocal, int localWorkers, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers), indexFirstLocal(indexFirstLocal) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	template <typename T>
	CollectorAdapter(ff_comb_t<Tin, T, Tout>* n, int indexFirstLocal, int localWorkers=1, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers), indexFirstLocal(indexFirstLocal) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
		internal_mi_transformer::registerCallback(cb,arg);
	}

	int svc_init() {
		if (this->n->isMultiOutput()) {
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_running(localWorkers + indexFirstLocal);
		}
		return n->svc_init();
	}

	void svc_end(){this->n->svc_end();}
	
	void * svc(void* in) {
        Tin * task;
        ssize_t channel;

        // if the results come from the "square box", it is a result from a remote workers so i have to read from which worker it come from 
		if (get_channel_id() == 0){
            ResultWrapper<Tin> * tw = reinterpret_cast<ResultWrapper<Tin>*>(in);
            task = tw->result;
            channel = tw->source;
        } else { // the result come from a local worker, just pass it to collector and compute the right worker id
            task = reinterpret_cast<Tin*>(in);
            channel = get_channel_id() + indexFirstLocal - 1;
        }


		if (this->n->isMultiInput()) {
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_input_channelid(channel, true);
		}		
		return n->svc(task);
	}

	ff_node* getOriginal(){ return this->n;	}
};





#endif