#ifndef WRAPPER_H
#define WRAPPER_H

#include <iostream>
#include <type_traits>
#include <functional>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>

using namespace ff;

template<typename Tin, typename Tout = Tin>
struct DummyNode : public ff_node_t<Tin, Tout> {
    Tout* svc(Tin* in){ return nullptr;}
};

class Wrapper {
protected:
	int myID; // used to populate sender field of the header

public:
	Wrapper() : myID(-1){}
	void setMyId(int id){myID = id;}
};

/*
	Wrapper IN class
*/
class WrapperIN: public internal_mi_transformer {

private:
    int inchannels;	// number of input channels the wrapped node is supposed to have
public:


	WrapperIN(ff_node* n, int inchannels=1, bool cleanup=false): internal_mi_transformer(this, false), inchannels(inchannels) {this->n = n; this->cleanup= cleanup;}

	void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
		internal_mi_transformer::registerCallback(cb,arg);
	}

	int svc_init() {
		if (this->n->isMultiOutput()) {
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_running(inchannels);
		}
		return n->svc_init();
	}

	void svc_end(){this->n->svc_end();}
	
		void * svc(void* in) {
		message_t* msg = (message_t*)in;	   
		
		if (this->n->isMultiInput()) {
            int channelid = msg->sender; 
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_input_channelid(channelid, true);
		}
		void* inputData = this->n->deserializeF(msg->data);

		delete msg;	
		return n->svc(inputData);
	}

	ff_node* getOriginal(){ return this->n;	}
};

template<bool Serialization, typename Tin, typename Tout = Tin>
struct WrapperINCustom : public WrapperIN {
	WrapperINCustom() : WrapperIN(new DummyNode<Tin, Tout>(), 1, true){}

	void * svc(void* in) {
		message_t* msg = (message_t*)in;	   
		auto* out = new SMmessage_t(this->n->deserializeF(msg->data), msg->sender, msg->chid);
		delete msg;
		return out;
	}
};


/*
	Wrapper OUT class
*/

class WrapperOUT: public Wrapper, public internal_mo_transformer {
private:
    int outchannels; // number of output channels the wrapped node is supposed to have
	int defaultDestination;
public:
	
	WrapperOUT(ff_node* n, int outchannels=1, bool cleanup=false, int defaultDestination = -1): Wrapper(), internal_mo_transformer(this, false), outchannels(outchannels), defaultDestination(defaultDestination) {
		this->n = n;
		this->cleanup= cleanup;
		registerCallback(ff_send_out_to_cbk, this);

	}
	
	bool serialize(void* in, int id) {
		if ((void*)in > FF_TAG_MIN) return this->ff_send_out(in);
		
		message_t* msg = new message_t;

	
		this->n->serializeF(in, msg->data);
		msg->sender = get_my_id(); // da cambiare con qualcosa di reale!
		msg->chid   = id;

		return this->ff_send_out(msg);
	}

	void * svc(void* in) {
		void* out = n->svc(in);
		if (out > FF_TAG_MIN) return out;					
		serialize(out, defaultDestination);
		return GO_ON;
	}

	int svc_init() {
		if (this->n->isMultiOutput()) {
			ff_monode* mo = reinterpret_cast<ff_monode*>(this->n);
			mo->set_running(outchannels);
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
		return ((WrapperOUT*)obj)->serialize(task, id);
	}
};

template<typename Tin, typename Tout = Tin>
struct WrapperOUTCustom : public WrapperOUT {
	WrapperOUTCustom() : WrapperOUT(new DummyNode<Tin, Tout>(), 1, true){}
	
	void* svc(void* in){
		SMmessage_t * input = reinterpret_cast<SMmessage_t*>(in);
		Wrapper::myID = input->sender;
		this->serialize(input->task, input->dst);
		delete input;
		return this->GO_ON;
	}
};


/*
	Wrapper INOUT class
*/
class WrapperINOUT: public Wrapper, public internal_mi_transformer {

private:
    int inchannels;	// number of input channels the wrapped node is supposed to have
    int outchannels; // number of output channels the wrapped node is supposed to have
	int defaultDestination;

public:

	WrapperINOUT(ff_node* n, int inchannels=1, bool cleanup=false, int defaultDestination = -1): internal_mi_transformer(this, false), inchannels(inchannels), defaultDestination(defaultDestination){
		this->n       = n;
		this->cleanup = cleanup;
        registerCallback(ff_send_out_to_cbk, this);
	}

	void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
		internal_mi_transformer::registerCallback(cb,arg);
	}

    bool serialize(void* in, int id) {
		if ((void*)in > FF_TAG_MIN) return this->ff_send_out(in);
		
		message_t* msg = new message_t;

	
		this->n->serializeF(in, msg->data);
		msg->sender = get_my_id(); // da cambiare con qualcosa di reale!
		msg->chid   = id;

		return this->ff_send_out(msg);
	}

	int svc_init() {
		if (this->n->isMultiOutput()) {
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_running(inchannels);
		}
		return n->svc_init();
	}

	void svc_end(){this->n->svc_end();}
	
	void * svc(void* in) {
		message_t* msg = (message_t*)in;

		if (this->n->isMultiInput()) {
            int channelid = msg->sender; 
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_input_channelid(channelid, true);
		}

		void* out = n->svc(this->n->deserializeF(msg->data));
        delete msg;

        if (out > FF_TAG_MIN) return out;

        serialize(out, defaultDestination);
        return GO_ON;
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

	ff_node* getOriginal(){ return this->n;	}
	
    static inline bool ff_send_out_to_cbk(void* task, int id,
										  unsigned long retry,
										  unsigned long ticks, void* obj) {		
		return ((WrapperINOUT*)obj)->serialize(task, id);
	}
};

#endif
