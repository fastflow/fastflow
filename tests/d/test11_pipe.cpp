/*
 *
 *          host0                           host1
 *    ------------------             -------------------
 *   |                  |           |                   |
 *   |                  |  UNICAST  |                   |
 *   | Node0 --> Node1 -|---------- |-> Node2 --> Node3 |
 *   |                  |           |                   |
 *   |  (ff_pipeline)   |           |   (ff_pipeline)   |
 *    ------------------             -------------------
 *
 *
 *
 */

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
/* Other options are: zmqOnDemand and zmqBcast */
#define COMM  zmq1_1 

class Node0: public ff_node {
public:
    void * svc(void*) {
	printf("Node0 starting\n");
	for(long i=1;i<=10000;++i)
	    ff_send_out((void*)i);
	printf("Node0 exiting\n");
	return NULL;
    }
};

class Node1: public ff_dnode<COMM> {
protected:
    static void callback(void * e,void* ) {
	delete ((long*)e);
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
	ff_dnode<COMM>::init(name, address, 1, transp, SENDER, 0, callback);  

	printf("Node1 starting\n");
	return 0;
    }

    void * svc(void *task) {	
	printf("Node1 received %ld\n", (long)task);
	return (new long((long)task)); // the callback deallocates the data
    }

    void svc_end() {
	printf("Node1 svn_end\n");
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
	ff_dnode<COMM>::init(name, address, 1, transp, RECEIVER);
	printf("Node2 starting\n");
	return 0;
    }

    void * svc(void *task) {
	//printf("Node2 received %ld\n", *(long*)task);
	return (new long(*(long*)task));
    }

    // overriding the default prepare method
    void prepare(svector<msg_t*>*& v, size_t len,const int=-1) {
	msgv.clear();
	msgv.reserve(len);
	msgv.push_back(&msg);
	v=&msgv;
    }

    void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& ptr) {
	// potentially, I can receive multiple messages depending on 
	// the fact that the sender has serialized the output data 
	// in multiple parts (peraphs because the message is not 
	// contiguous in memory)
	//
	assert(vlen==1 && v[0]->size()==1); // in this example we have just 1 msg
        ptr = v[0]->operator[](0)->getData();
    }

protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
private:
    svector<msg_t*> msgv;
    msg_t msg;
    double t;
};

class Node3: public ff_node {
public:
    void *svc(void *task) {
	printf("Node3 received %ld\n", *(long*)task);
	delete ((long*)task);
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
    zmqTransport transport(0);
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
