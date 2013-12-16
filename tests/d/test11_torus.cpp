/*
 *
 *             host0                           host1
 *       --------------------   UNICAST   -------------------
 *      |                    |    or     |                   |
 *      |                    |  ONDEMAND |                   |
 *    --|-> Node0 --> Node1 -|---------- |-> Node2 --> Node3 |--
 *   |  |                    |           |                   |  |
 *   |  |    (ff_pipeline)   |           |   (ff_pipeline)   |  |
 *   |   --------------------             -------------------   |
 *   |                          UNICAST                         |
 *    ----------------------------------------------------------
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

#define NTASKS 10000
#define COMM   zmq1_1

class Node0: public ff_dnode<zmq1_1> {
public:
    
    Node0(const std::string& name, const std::string& address, zmq1_1::TransportImpl* const transp):
	ntasks(NTASKS), name(name),address(address),transp(transp) { }

    int svc_init() {
	ff_dnode<zmq1_1>::init(name, address,1, transp,RECEIVER);
	ff::ffTime(START_TIME);
	return 0;
    }

    void * svc(void* task) {
	if (task==NULL) {
	    for(unsigned long i=1;i<=ntasks;++i)
		ff_send_out((void*)i);
	    return GO_ON;
	}
	printf("Node0 received %ld\n", *(long*)task);

	if (--ntasks == 0) return NULL;
	return GO_ON;
    }

    void svc_end() {
	printf("Time= %f ms\n", ff::ffTime(STOP_TIME));
    }


    // overriding the default prepare and unmarshall methods
    void prepare(svector<msg_t*>*& v, size_t len,const int=-1) {
	msgv.clear();
	msgv.reserve(len);
	msgv.push_back(&msg);
	v=&msgv;
    }
    void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& ptr) {
	assert(vlen==1 && v[0]->size()==1); // in this example we have just 1 msg
        ptr = v[0]->operator[](0)->getData();
    }


private:
    svector<msg_t*> msgv;
    msg_t msg;

private:
    unsigned long ntasks;
    const std::string name;
    const std::string address;
    zmq1_1::TransportImpl* const transp;
};

class Node1: public ff_dnode<COMM> { 
protected:
    static void callback(void * e,void* ) {
	delete ((long*)e);
    }
public:
    
    Node1(const std::string& name, const std::string& address, zmq1_1::TransportImpl* const transp):
	name(name), address(address), transp(transp) {}

    int svc_init() {
	ff_dnode<COMM>::init(name, address,1, transp,SENDER, 0, callback);  	
	return 0;
    }

    void * svc(void *task) {	
	return (new long((long)task)); // the callback deallocates the data
    }

private:
    const std::string name;
    const std::string address;
    COMM::TransportImpl* const transp;
};

class Node2: public ff_dnode<COMM> { 
public:
    typedef zmqTransport::msg_t msg_t;

    Node2(const std::string& name, const std::string& address,zmq1_1::TransportImpl* const transp):
	name(name),address(address),transp(transp) {}

    int svc_init() {
	ff_dnode<COMM>::init(name, address,1, transp, RECEIVER, 0);  
	return 0;
    }

    void * svc(void *task) {
	return (new long(*(long*)task));
    }

    // overriding the default prepare and unmarshall methods
    void prepare(svector<msg_t*>*& v, size_t len,const int=-1) {
	msgv.clear();
	msgv.reserve(len);
	msgv.push_back(&msg);
	v=&msgv;
    }
    void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& ptr) {
	assert(vlen==1 && v[0]->size()==1); // in this example we have just 1 msg
        ptr = v[0]->operator[](0)->getData();
    }

private:
    svector<msg_t*> msgv;
    msg_t msg;
    const std::string name;
    const std::string address;
    COMM::TransportImpl* const transp;
};

class Node3: public ff_dnode<zmq1_1> { 
protected:
    static void callback(void * e,void*) {
	delete ((long*)e);
    }
public:
    Node3(const std::string& name, const std::string& address, zmq1_1::TransportImpl* const transp):
	name(name),address(address),transp(transp) {}

    int svc_init() {
	ff_dnode<zmq1_1>::init(name, address,1, transp, SENDER, 0, callback);  
	return 0;
    }

    void *svc(void *task) {
	return task;
    }
private:
    const std::string name;
    const std::string address;
    zmq1_1::TransportImpl* const transp;
};



int main(int argc, char * argv[]) {    
    if (argc != 6) {
	std::cerr << "use: " << argv[0] << " name1 name2 1|0 host1:port1 host2:port2\n";
	return -1;
    }
    
    char * name1 = argv[1];
    char * name2 = argv[2];
    char * P = argv[3];         // 1 producer 0 consumer
    char * address1 = argv[4];  // no check
    char * address2 = argv[5];  // no check

    // creates the network using 0mq as transport layer
    zmqTransport transport(0);
    if (transport.initTransport()<0) abort();

    if (atoi(P)) {
	Node0 * n0 = new Node0(name2, address2, &transport);
	Node1 * n1 = new Node1(name1, address1, &transport);
	n0->skipfirstpop(true);

	ff_pipeline pipe(false, NTASKS); 
	pipe.add_stage(n0);
	pipe.add_stage(n1);
	pipe.run_and_wait_end();

	delete n0;
	delete n1;
    } else {
	Node2 * n2 = new Node2(name1, address1, &transport);
	Node3 * n3 = new Node3(name2, address2, &transport);
	
	ff_pipeline pipe(false, NTASKS); 
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
