/*
 *
 *         host0                  host1
 *       ---------             ----------
 *      |         |           |          |
 *      |         |  UNICAST  |          |
 *      |  Node1 -|---------- |-> Node2  |
 *      |         |           |          |  
 *       ---------             ----------
 *
 *
 * This simple test measures the bandwidith (Mb/s) of sending 'size'
 * bytes from host0 to host1. 
 *
 *   ((COUNT/ Node2 computation time IN SECS)*size*8)/1000000
 * 
 */
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <string>

#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>
#include <ff/dnode.hpp>
#include <ff/pipeline.hpp>
#include <ff/utils.hpp>

const long int COUNT=1000;

using namespace ff;

#define COMM zmqOnDemand

class Node1: public ff_dnode<COMM> {
protected:
    static void callback(void * e,void*) {
	delete [] ((char*)e);
    }
public:    
    Node1(const unsigned size, const std::string& name, const std::string& address, zmqTransport* const transp):
	size(size),name(name), address(address), transp(transp) {}

    int svc_init() {
	ff_dnode<COMM>::init(name, address, 1,transp,true, 0); //, callback);
	printf("Node1 start\n");
	return 0;
    }
    void * svc(void*) {
	char* data=new char[size];
	for(int i=0;i<COUNT;++i) {
	    //char* data=new char[size];
	    ff_send_out((void*)data);
	}
	return NULL;
    }
    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        struct iovec iov={ptr,size*sizeof(char)};
        v.push_back(iov);
	setCallbackArg(NULL);
    }    
private:
    const unsigned size;
    const std::string name;
    const std::string address;
    zmqTransport* const transp;
};

    
class Node2: public ff_dnode<COMM> {
public:
    typedef zmqTransport::msg_t msg_t;
    
    Node2(const unsigned size, const std::string& name, const std::string& address, zmqTransport* const transp):
	size(size),name(name),address(address),transp(transp) {}
    
    int svc_init() {
	ff_dnode<COMM>::init(name,address, 1, transp, false, 0);  
	printf("Node2 start\n");
	ff::ffTime(START_TIME);
	return 0;
    }
    void * svc(void *task) { 
	COMM::TransportImpl::msg_t* msg=(COMM::TransportImpl::msg_t*)task;
	assert(size == msg->size());
	delete msg;
	return GO_ON;
    }
    void svc_end() {
	ff::ffTime(STOP_TIME);
	printf("Time = %f ms\n", ff::ffTime(GET_TIME));
	printf("Bandwidth = %.3f Mb/s\n", 
	       (((double)(COUNT*1000) /(double)ff::ffTime(GET_TIME))*size*8)/1000000.0);
    }
    void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
	// returns the msg
	task = v[0]->operator[](0); 
        delete v[0];
    }
private:
    const unsigned size;
    const std::string name;
    const std::string address;
    zmqTransport* const transp;
};
    
    
int main(int argc, char * argv[]) {    
    if (argc != 4) {
	std::cerr << "use: " << argv[0] << " 1|0 host:port data-size(bytes)\n";
	return -1;
    }
    
    char * P = argv[1];         // 1 producer 0 consumer
    char * address1 = argv[2];  // no check, this is the address of the producer
    unsigned size = atoi(argv[3]);

    // creates the network using 0mq as transport layer
    zmqTransport transport(atoi(P));
    if (transport.initTransport()<0) abort();

    if (atoi(P)) {
	Node1* n1 = new Node1(size, "A", address1, &transport);
	n1->run();
	n1->wait();
	delete n1;
    } else {
	Node2 * n2 = new Node2(size, "A", address1, &transport);
	n2->run();
	n2->wait();
	delete n2;
    }
    
    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
