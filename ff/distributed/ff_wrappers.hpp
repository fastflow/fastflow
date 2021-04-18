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


/*
	Wrapper IN class
*/
template<bool Serialization, typename Tin, typename Tout = Tin>
class WrapperIN: public internal_mi_transformer {

private:
    int inchannels;	// number of input channels the wrapped node is supposed to have
public:
	using ff_deserialize_F_t = 	std::function<std::add_pointer_t<Tin>(char*,size_t)>;

	WrapperIN(ff_node_t<Tin, Tout>* n, int inchannels=1, bool cleanup=false, const ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; })): internal_mi_transformer(this, false), inchannels(inchannels), finalizer_(finalizer) {
		this->n       = n;
		this->cleanup = cleanup;
	}
	WrapperIN(ff_node* n, int inchannels=1, bool cleanup=false, const ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; })): internal_mi_transformer(this, false), inchannels(inchannels), finalizer_(finalizer) {
		this->n       = n;
		this->cleanup = cleanup;
	}
	WrapperIN(ff_minode_t<Tin, Tout>* n, int inchannels, bool cleanup=false, const ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; })): internal_mi_transformer(this, false), inchannels(inchannels), finalizer_(finalizer) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	WrapperIN(ff_monode_t<Tin, Tout>* n, int inchannels, bool cleanup=false, const ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; })): internal_mi_transformer(this, false), inchannels(inchannels), finalizer_(finalizer) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	template <typename T>
	WrapperIN(ff_comb_t<Tin, T, Tout>* n, int inchannels=1, bool cleanup=false, const ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; })): internal_mi_transformer(this, false), inchannels(inchannels), finalizer_(finalizer) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
		internal_mi_transformer::registerCallback(cb,arg);
	}

	Tin* deserialize(void* buffer) {
		message_t* msg = (message_t*)buffer;
		Tin* data = nullptr;
		if constexpr (Serialization) {
			// deserialize the buffer into a heap allocated buffer		
			
			std::istream iss(&msg->data);
			cereal::PortableBinaryInputArchive iarchive(iss);

			data = new Tin;
			iarchive >> *data;
				
		} else {
			// deserialize the buffer into a heap allocated buffer		
			
			msg->data.doNotCleanup(); // to not delete the area of memory during the delete of the message_t wrapper
			data = finalizer_(msg->data.getPtr(), msg->data.getLen());
		}

		delete msg;	
		return data;
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
		int channelid = msg->sender; 
		if (this->n->isMultiInput()) {
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_input_channelid(channelid, true);
		}		
		return n->svc(deserialize(in));
	}

	ff_node* getOriginal(){ return this->n;	}

	ff_deserialize_F_t finalizer_;
	ff_deserialize_F_t getFinalizer() {return this->finalizer_;}
};


/*
	Wrapper OUT class
*/
template<bool Serialization, typename Tin, typename Tout = Tin>
class WrapperOUT: public internal_mo_transformer {
private:
    int outchannels; // number of output channels the wrapped node is supposed to have
public:

	typedef Tout T_out;
	using ff_serialize_F_t   = std::function<std::pair<char*,size_t>(std::add_pointer_t<Tout>)>;
	
	WrapperOUT(ff_node_t<Tin, Tout>* n, int outchannels=1, bool cleanup=false, ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): internal_mo_transformer(this, false), outchannels(outchannels), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}
	WrapperOUT(ff_node* n, int outchannels=1, bool cleanup=false, ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): internal_mo_transformer(this, false), outchannels(outchannels), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}
	
	WrapperOUT(ff_monode_t<Tin, Tout>* n, int outchannels=1, bool cleanup=false, ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): internal_mo_transformer(this, false), outchannels(outchannels), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}

	WrapperOUT(ff_minode_t<Tin, Tout>* n, int outchannels=1, bool cleanup=false, ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): internal_mo_transformer(this, false), outchannels(outchannels), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}

	template <typename T>
	WrapperOUT(ff_comb_t<Tin, T, Tout>* n, int outchannels=1, bool cleanup=false, ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): internal_mo_transformer(this, false), outchannels(outchannels), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}
	
	bool serialize(Tout* in, int id) {
		if ((void*)in > FF_TAG_MIN) return this->ff_send_out(in);
		
		message_t* msg = nullptr;

		if constexpr (Serialization){
			msg = new message_t;
			assert(msg);
			msg->sender = this->n->get_my_id();
			msg->chid   = id;
			std::ostream oss(&msg->data);
			cereal::PortableBinaryOutputArchive oarchive(oss);
			oarchive << *in;
			delete in;
		} else {
			std::pair<char*, size_t> raw_data = transform_(in); 

			msg = new message_t(raw_data.first, raw_data.second, true);
			assert(msg);
			msg->sender = this->n->get_my_id();
			msg->chid   = id;
		}

		return this->ff_send_out(msg);

	}

	void * svc(void* in) {
		void* out = n->svc(in);
		if (out > FF_TAG_MIN) return out;					
		serialize(reinterpret_cast<Tout*>(out), -1);
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


	ff_serialize_F_t transform_;
	ff_serialize_F_t getTransform() { return this->transform_;}
	
	ff::ff_node* getOriginal(){return this->n;}
	
	static inline bool ff_send_out_to_cbk(void* task, int id,
										  unsigned long retry,
										  unsigned long ticks, void* obj) {		
		return ((WrapperOUT*)obj)->serialize(reinterpret_cast<WrapperOUT::T_out*>(task), id);
	}
};


/*
	Wrapper INOUT class
*/
template<bool SerializationIN, bool SerializationOUT, typename Tin, typename Tout = Tin>
class WrapperINOUT: public internal_mi_transformer {

private:
    int inchannels;	// number of input channels the wrapped node is supposed to have
    int outchannels; // number of output channels the wrapped node is supposed to have
public:

    typedef Tout T_out;
	using ff_serialize_F_t   = std::function<std::pair<char*,size_t>(std::add_pointer_t<Tout>)>;
	using ff_deserialize_F_t = 	std::function<std::add_pointer_t<Tin>(char*,size_t)>;

	WrapperINOUT(ff_node_t<Tin, Tout>* n, int inchannels=1, bool cleanup=false, ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; }), ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): internal_mi_transformer(this, false), inchannels(inchannels), finalizer_(finalizer), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
        registerCallback(ff_send_out_to_cbk, this);
	}	

	WrapperINOUT(ff_minode_t<Tin, Tout>* n, int inchannels=1, bool cleanup=false, ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; }), ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): internal_mi_transformer(this, false), inchannels(inchannels), finalizer_(finalizer), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
        registerCallback(ff_send_out_to_cbk, this);
	}

	WrapperINOUT(ff_monode_t<Tin, Tout>* n, int inchannels=1, bool cleanup=false, ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; }), ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): internal_mi_transformer(this, false), inchannels(inchannels), finalizer_(finalizer), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
        registerCallback(ff_send_out_to_cbk, this);
	}

	template <typename T>
	WrapperINOUT(ff_comb_t<Tin, T, Tout>* n, int inchannels=1, bool cleanup=false, ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; }), ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): internal_mi_transformer(this, false), inchannels(inchannels), finalizer_(finalizer), transform_(transform){
		this->n       = n;
		this->cleanup = cleanup;
        registerCallback(ff_send_out_to_cbk, this);
	}

	void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
		internal_mi_transformer::registerCallback(cb,arg);
	}

	
	Tin* deserialize(void* buffer) {
		message_t* msg = (message_t*)buffer;
		Tin* data = nullptr;
		if constexpr (SerializationIN) {
			// deserialize the buffer into a heap allocated buffer		
			
			std::istream iss(&msg->data);
			cereal::PortableBinaryInputArchive iarchive(iss);

			data = new Tin;
			iarchive >> *data;
				
		} else {
			// deserialize the buffer into a heap allocated buffer		
			
			msg->data.doNotCleanup(); // to not delete the area of memory during the delete of the message_t wrapper
			data = finalizer_(msg->data.getPtr(), msg->data.getLen());
		}

		delete msg;	
		return data;
	}

    bool serialize(Tout* in, int id) {
		if ((void*)in > FF_TAG_MIN) return ff_node::ff_send_out(in);
		
		message_t* msg = nullptr;

		if constexpr (SerializationOUT){
			msg = new message_t;
			assert(msg);
			msg->sender = this->get_my_id();
			msg->chid   = id;
			std::ostream oss(&msg->data);
			cereal::PortableBinaryOutputArchive oarchive(oss);
			oarchive << *in;
			delete in;
		} else {
			std::pair<char*, size_t> raw_data = transform_(in); 

			msg = new message_t(raw_data.first, raw_data.second, true);
			assert(msg);
			msg->sender = this->get_my_id();
			msg->chid   = id;
		}

		return ff_node::ff_send_out(msg);
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
		int channelid = msg->sender; 
		if (this->n->isMultiInput()) {
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_input_channelid(channelid, true);
		}		

		void* out = n->svc(deserialize(in));
        if (out > FF_TAG_MIN) return out;
        serialize(reinterpret_cast<Tout*>(out), -1);
        return GO_ON;
	}
	
	ff_deserialize_F_t finalizer_;
	std::function<std::pair<char*,size_t>(Tout*)> transform_;

	ff_node* getOriginal(){ return this->n;	}
	
    static inline bool ff_send_out_to_cbk(void* task, int id,
										  unsigned long retry,
										  unsigned long ticks, void* obj) {		
		return ((WrapperINOUT*)obj)->serialize(reinterpret_cast<WrapperINOUT::T_out*>(task), id);
	}
};

#endif
