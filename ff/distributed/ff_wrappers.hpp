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
    Tout* svc(Tin* i){}
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
template<bool Serialization, typename Tin, typename Tout = Tin>
class WrapperIN: public internal_mi_transformer {

private:
    int inchannels;	// number of input channels the wrapped node is supposed to have
public:
	using ff_deserialize_F_t = 	std::function<std::add_pointer_t<Tin>(char*,size_t)>;

	WrapperIN(ff_node_t<Tin, Tout>* n, int inchannels=1, bool cleanup=false, const ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; })): internal_mi_transformer(n, cleanup), inchannels(inchannels), finalizer_(finalizer) {}

	WrapperIN(ff_node* n, int inchannels=1, bool cleanup=false, const ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; })): internal_mi_transformer(n, cleanup), inchannels(inchannels), finalizer_(finalizer) {}

	WrapperIN(ff_minode_t<Tin, Tout>* n, int inchannels, bool cleanup=false, const ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; })): internal_mi_transformer(n, cleanup), inchannels(inchannels), finalizer_(finalizer) {}

	WrapperIN(ff_monode_t<Tin, Tout>* n, int inchannels, bool cleanup=false, const ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; })): internal_mi_transformer(n, cleanup), inchannels(inchannels), finalizer_(finalizer) {}

	template <typename T>
	WrapperIN(ff_comb_t<Tin, T, Tout>* n, int inchannels=1, bool cleanup=false, const ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; })): internal_mi_transformer(n, cleanup), inchannels(inchannels), finalizer_(finalizer) {}

	void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
		internal_mi_transformer::registerCallback(cb,arg);
	}

	Tin* deserialize(dataBuffer& dbuffer) {
		Tin* data = nullptr;
		if constexpr (Serialization) {
			// deserialize the buffer into a heap allocated buffer		
			
			std::istream iss(&dbuffer);
			cereal::PortableBinaryInputArchive iarchive(iss);

			data = new Tin;
			iarchive >> *data;
				
		} else {
			// deserialize the buffer into a heap allocated buffer		
			
			dbuffer.doNotCleanup(); // to not delete the area of memory during the delete of the message_t wrapper
			data = finalizer_(dbuffer.getPtr(), dbuffer.getLen());
		}

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
		Tin* inputData = deserialize(msg->data);
		delete msg;	
		return n->svc(inputData);
	}

	ff_node* getOriginal(){ return this->n;	}

	ff_deserialize_F_t finalizer_;
	ff_deserialize_F_t getFinalizer() {return this->finalizer_;}
};

template<bool Serialization, typename Tin, typename Tout = Tin>
struct WrapperINCustom : public WrapperIN<Serialization, Tin, Tout> {
	WrapperINCustom() : WrapperIN<Serialization, Tin, Tout>(new DummyNode<Tin, Tout>(), 1, true){}

	void * svc(void* in) {
		message_t* msg = (message_t*)in;	   
		return new SMmessage_t(this->deserialize(msg->data), msg->sender, msg->chid);
	}
};


/*
	Wrapper OUT class
*/
template<bool Serialization, typename Tin, typename Tout = Tin>
class WrapperOUT: public Wrapper, public internal_mo_transformer {
private:
    int outchannels; // number of output channels the wrapped node is supposed to have
	int defaultDestination;
public:

	typedef Tout T_out;
	using ff_serialize_F_t   = std::function<std::pair<char*,size_t>(std::add_pointer_t<Tout>)>;
	
	WrapperOUT(ff_node_t<Tin, Tout>* n, int outchannels=1, bool cleanup=false, int defaultDestination = -1, ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): Wrapper(), internal_mo_transformer(n, cleanup), outchannels(outchannels), defaultDestination(defaultDestination), transform_(transform) {
		registerCallback(ff_send_out_to_cbk, this);
	}
	
	WrapperOUT(ff_node* n, int outchannels=1, bool cleanup=false, int defaultDestination = -1, ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): Wrapper(), internal_mo_transformer(n, cleanup), outchannels(outchannels), defaultDestination(defaultDestination), transform_(transform) {
		registerCallback(ff_send_out_to_cbk, this); //maybe useless??
	}
	
	WrapperOUT(ff_monode_t<Tin, Tout>* n, int outchannels=1, bool cleanup=false, int defaultDestination = -1, ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): Wrapper(), internal_mo_transformer(n, cleanup), outchannels(outchannels), defaultDestination(defaultDestination), transform_(transform) {
		registerCallback(ff_send_out_to_cbk, this); // maybe useless??
	}

	WrapperOUT(ff_minode_t<Tin, Tout>* n, int outchannels=1, bool cleanup=false, int defaultDestination = -1, ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): Wrapper(), internal_mo_transformer(n, cleanup), outchannels(outchannels), defaultDestination(defaultDestination), transform_(transform) {
		registerCallback(ff_send_out_to_cbk, this);
	}

	template <typename T>
	WrapperOUT(ff_comb_t<Tin, T, Tout>* n, int outchannels=1, bool cleanup=false, int defaultDestination = -1, ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): Wrapper(), internal_mo_transformer(this, false), outchannels(outchannels), defaultDestination(defaultDestination), transform_(transform) {
		registerCallback(ff_send_out_to_cbk, this);
	}
	
	bool serialize(Tout* in, int id) {
		if ((void*)in > FF_TAG_MIN) return this->ff_send_out(in);
		
		message_t* msg = nullptr;

		if constexpr (Serialization){
			msg = new message_t;
			assert(msg);
			msg->sender = this->myID;
			msg->chid   = id;
			std::ostream oss(&msg->data);
			cereal::PortableBinaryOutputArchive oarchive(oss);
			oarchive << *in;
			delete in;
		} else {
			std::pair<char*, size_t> raw_data = transform_(in); 

			msg = new message_t(raw_data.first, raw_data.second, true);
			assert(msg);
			msg->sender = this->myID;
			msg->chid   = id;
		}

		return this->ff_send_out(msg);

	}

	void * svc(void* in) {
		void* out = n->svc(in);
		if (out > FF_TAG_MIN) return out;					
		serialize(reinterpret_cast<Tout*>(out), defaultDestination);
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

template<bool Serialization, typename Tin, typename Tout = Tin>
struct WrapperOUTCustom : public WrapperOUT<Serialization, Tin, Tout> {
	WrapperOUTCustom() : WrapperOUT<Serialization, Tin, Tout>(new DummyNode<Tin, Tout>(), 1, true){}
	
	void* svc(void* in){
		SMmessage_t * input = reinterpret_cast<SMmessage_t*>(in);
		Wrapper::myID = input->sender;
		this->serialize(reinterpret_cast<Tout*>(input->task), input->dst);
		return this->GO_ON;
	}
};


/*
	Wrapper INOUT class
*/
template<bool SerializationIN, bool SerializationOUT, typename Tin, typename Tout = Tin>
class WrapperINOUT: public Wrapper, public internal_mi_transformer {

private:
    int inchannels;	// number of input channels the wrapped node is supposed to have
    int outchannels; // number of output channels the wrapped node is supposed to have
	int defaultDestination;

public:

    typedef Tout T_out;
	using ff_serialize_F_t   = std::function<std::pair<char*,size_t>(std::add_pointer_t<Tout>)>;
	using ff_deserialize_F_t = 	std::function<std::add_pointer_t<Tin>(char*,size_t)>;

	WrapperINOUT(ff_node_t<Tin, Tout>* n, int inchannels=1, bool cleanup=false, int defaultDestination = -1, ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; }), ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): Wrapper(), internal_mi_transformer(this, false), inchannels(inchannels), defaultDestination(defaultDestination), finalizer_(finalizer), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
        registerCallback(ff_send_out_to_cbk, this);
	}	

	WrapperINOUT(ff_minode_t<Tin, Tout>* n, int inchannels=1, bool cleanup=false, int defaultDestination = -1, ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; }), ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): Wrapper(), internal_mi_transformer(this, false), inchannels(inchannels), defaultDestination(defaultDestination), finalizer_(finalizer), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
        registerCallback(ff_send_out_to_cbk, this);
	}

	WrapperINOUT(ff_monode_t<Tin, Tout>* n, int inchannels=1, bool cleanup=false, int defaultDestination = -1, ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; }), ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): Wrapper(), internal_mi_transformer(this, false), inchannels(inchannels), defaultDestination(defaultDestination), finalizer_(finalizer), transform_(transform) {
		this->n       = n;
		this->cleanup = cleanup;
        registerCallback(ff_send_out_to_cbk, this);
	}

	template <typename T>
	WrapperINOUT(ff_comb_t<Tin, T, Tout>* n, int inchannels=1, bool cleanup=false, int defaultDestination = -1, ff_deserialize_F_t finalizer=([](char* p, size_t){ return new (p) Tin; }), ff_serialize_F_t transform=([](Tout* in){return std::make_pair((char*)in, sizeof(Tout));})): Wrapper(), internal_mi_transformer(this, false), inchannels(inchannels), defaultDestination(defaultDestination), finalizer_(finalizer), transform_(transform){
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
			msg->sender = this->myID;
			msg->chid   = id;
			std::ostream oss(&msg->data);
			cereal::PortableBinaryOutputArchive oarchive(oss);
			oarchive << *in;
			delete in;
		} else {
			std::pair<char*, size_t> raw_data = transform_(in); 

			msg = new message_t(raw_data.first, raw_data.second, true);
			assert(msg);
			msg->sender = this->myID;
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
        serialize(reinterpret_cast<Tout*>(out), defaultDestination);
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
