#include <sys/uio.h>
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdint.h>

#include <ff/node.hpp>
#include <ff/svector.hpp>
#include <ff/dnode.hpp>
#include <ff/pipeline.hpp>
#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>

using namespace ff;

#define COMM zmqBcast

// my string type
struct mystring_t {
    mystring_t(int l, char* str):length(l),str(str) {}
    unsigned  length;
    char    * str;
};

class Node0: public ff_node {
public:
    void * svc(void*) {
	printf("Node0 starting\n");
	ff_send_out((void*)GO_ON);
	printf("Node0 exiting\n");
	return NULL;
    }
};

class Node1: public ff_dnode<COMM> {
protected:
    static void callback(void * e,void*) {
	static int flag=0;
	if (flag==0) {
	    mystring_t* s = (mystring_t*)e;
	    delete s;
	    flag=1;
	    return;
	}
	char* s = (char*)e;
	delete [] s;
	flag=0;
    }

public:
    typedef COMM::TransportImpl        transport_t;

    Node1(const std::string& name, const std::string& address, transport_t* const transp):
	name(name),address(address),transp(transp) {
    }

    // initializes dnode
    int svc_init() {
	// the callback will be called as soon as the output message is no 
	// longer in use by the transport layer
	ff_dnode<COMM>::init(name, address,1, transp, SENDER, 0, callback);  

	printf("Node1 starting\n");
	return 0;
    }

    void * svc(void *task) {	
	printf("Node1 started\n");
	
	char * s1 = new char[12+1];
	strncpy(s1, "Hello World!", 12+1);
	mystring_t* s = new mystring_t(12,s1);
	ff_send_out(s);

	char * s2 = new char[32+1];
	strncpy(s2, "This is just a very simple test.", 32+1);
	mystring_t* k = new mystring_t(32,s2);
	ff_send_out(k);

	return NULL;
    }

    void svc_end() {
	printf("Node1 ending\n");
    }

    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        mystring_t* p = static_cast<mystring_t*>(ptr);
        struct iovec iov={ptr,sizeof(mystring_t)};
        v.push_back(iov);
	iov.iov_base = p->str;
	iov.iov_len  = p->length+1;
	v.push_back(iov);
    }    


protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};

class Node2: public ff_dnode<COMM> {
public:
    typedef zmqTransport::msg_t    msg_t;
    typedef COMM::TransportImpl  transport_t;

    Node2(const std::string& name, const std::string& address, zmqTransport* const transp):
	name(name),address(address),transp(transp) {
    }

    int svc_init() {
	// initializes dnode
	return ff_dnode<COMM>::init(name, address, 1, transp, RECEIVER);
    }

    void * svc(void *task) {
	mystring_t* s = (mystring_t*)task;
	printf("Node2 received %s\n", s->str);
	return task;
    }

    // overriding the default prepare method
    void prepare(svector<msg_t*>*& v, size_t len,const int=-1) {
	assert(len==2);
	msgv.clear();
	msgv.reserve(len);
	msgv.push_back(&msg1);
	msgv.push_back(&msg2);
	v=&msgv;
    }
#if 0
    void unmarshalling(svector<msg_t*>* const v[],const int vlen,
		       void *& task) {
        assert(vlen==1 && v[0]->size()==2); 
	mystring_t* p =static_cast<mystring_t*>(v[0]->operator[](0)->getData());
        p->str = static_cast<char*>(v[0]->operator[](1)->getData());
	assert(strlen(p->str)== p->length);
	task=p;
    }
#endif

    void unmarshalling(svector<msg_t*>* const v[],const int vlen,
		       void *& task) {
        assert(vlen==1 && v[0]->size()==2); 
	mystring_t* p =static_cast<mystring_t*>(v[0]->operator[](0)->getData());
        p->str = static_cast<char*>(v[0]->operator[](1)->getData());
	assert(strlen(p->str)== p->length);
	task=p;
    }

protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
private:
    svector<msg_t*> msgv;
    msg_t msg1;
    msg_t msg2;
};

class Node3: public ff_node {
public:
    void *svc(void *task) {
	return GO_ON;
    }
};



int main(int argc, char * argv[]) {    
    if (argc != 4) {
	std::cerr << "use: " << argv[0] << " name 1|0 host:port\n";
	return -1;
    }
    
    char * name = argv[1];
    char * P = argv[2];        // 1 producer 0 consumer
    char * address = argv[3];  // no check

    // creates the network using 0mq as transport layer
    zmqTransport transport(atoi(P));
    if (transport.initTransport()<0) abort();

    if (atoi(P)) {
	Node0 * n0 = new Node0;
	Node1 * n1 = new Node1(name, address, &transport);

	ff_pipeline pipe;
	pipe.add_stage(n0);
	pipe.add_stage(n1);
	pipe.run_and_wait_end();

	delete n0;
	delete n1;
    } else {
	Node2 * n2 = new Node2(name, address, &transport);
	Node3 * n3 = new Node3;
	
	ff_pipeline pipe;
	pipe.add_stage(n2);
	pipe.add_stage(n3);
	pipe.run_and_wait_end();
	delete n2;
	delete n3;
    }
    // TODO: shutdown protocol

    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
