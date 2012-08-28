/*
 *
 *             host0                           host1
 *       --------------------             -------------------
 *      |                    |           |                   |
 *      |                    |  UNICAST  |                   |
 *    --|-> Node0 --> Node1 -|---------- |-> Node2 --> Node3 |--
 *   |  |                    |           |                   |  |
 *   |  |    (ff_pipeline)   |           |   (ff_pipeline)   |  |
 *   |   --------------------             -------------------   |
 *   |                          UNICAST                         |
 *    ----------------------------------------------------------
 *
 *
 * This simple test measures the latency of sending one long integer
 * from host0 to host1. The Node0 sends ROUNDTRIP_COUNT integers out
 * and wait to receive them back from Node3. 
 * When it receives the last one in input, then sends out the EOS. 
 * The latency is calculated as:
 *   Node0 computation time / (ROUNDTRIP_COUNT*2)
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

const long int ROUNDTRIP_COUNT=1000000;

using namespace ff;

class Node0: public ff_dnode<zmq1_1> {
public:    
  Node0(const std::string& name, const std::string& address, zmqTransport* const transp): cnt(0),sum(0),
	name(name), address(address), transp(transp) {}

    int svc_init() {
	ff_dnode<zmq1_1>::init(name, address, 1,transp,false, 0);  
	printf("Node0 start\n");
	ff::ffTime(START_TIME);
	return 0;
    }
    
    void * svc(void* task) {
      if (task==NULL) { 
	ff_send_out(new long(++cnt)); // will be deleted in Node1
	return GO_ON;
      }
      if (cnt > ROUNDTRIP_COUNT) return NULL; // generates EOS
      sum+=*(long*)task; // just to check the final result
      ff_send_out(new long(++cnt)); // will be deleted in Node1
      return GO_ON;
    }
    void svc_end() {
	ff::ffTime(STOP_TIME);
	printf("The sum of %ld integers is %ld\n", ROUNDTRIP_COUNT, sum);
	printf("Latency = %f ms\n", (ff::ffTime(GET_TIME)/(ROUNDTRIP_COUNT*2)));
    }
private:
    long cnt,sum;
    const std::string name;
    const std::string address;
    zmqTransport* const transp;
};


class Node1: public ff_dnode<zmq1_1> {
protected:
    static void callback(void * e,void*) {
	delete ((long*)e);
    }
public:    
    Node1(const std::string& name, const std::string& address, zmqTransport* const transp):
	name(name), address(address), transp(transp) {}

    int svc_init() {
      ff_dnode<zmq1_1>::init(name, address, 1,transp,true, 0, callback);  
      printf("Node1 start\n");
      return 0;
    }
    void * svc(void * task) { return task;}
private:
    const std::string name;
    const std::string address;
    zmqTransport* const transp;
};

    
class Node2: public ff_dnode<zmq1_1> {
public:
    typedef zmqTransport::msg_t msg_t;
    
    Node2(const std::string& name, const std::string& address, zmqTransport* const transp):
	name(name),address(address),transp(transp) {}
    
    int svc_init() {
	ff_dnode<zmq1_1>::init(name,address, 1, transp, false, 0);  
	printf("Node2 start\n");
	return 0;
    }
    void * svc(void *task) { return task;}

    void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
      // returns the msg
      // the message (allocated by the run-time) will be deleted in Node3
        task = v[0]->operator[](0); 
        delete v[0];
    }
private:
    const std::string name;
    const std::string address;
    zmqTransport* const transp;
};
    

class Node3: public ff_dnode<zmq1_1> {
protected:
    static void callback(void * e,void* arg) {
	delete (zmq1_1::TransportImpl::msg_t*)arg;
    }
public:
    typedef zmqTransport::msg_t msg_t;
    
    Node3(const std::string& name, const std::string& address, zmqTransport* const transp):
	name(name),address(address),transp(transp) {}
    
    int svc_init() {
	ff_dnode<zmq1_1>::init(name,address, 1, transp, true, 0, callback);  
	printf("Node3 start\n");
	return 0;
    }    
    void* svc(void *task) { return task;}

  void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
    // just send the data not the message
    struct iovec iov={((zmq1_1::TransportImpl::msg_t*)ptr)->getData(),sizeof(void*)};
    v.push_back(iov);
    setCallbackArg(ptr); // set the pointer for the callback
  }    

private:
    const std::string name;
    const std::string address;
    zmqTransport* const transp;
};

    
int main(int argc, char * argv[]) {    
    if (argc != 4) {
	std::cerr << "use: " << argv[0] << " 1|0 producer-host:port1 consumer-host:port2\n";
	return -1;
    }
    
    char * P = argv[1];         // 1 producer 0 consumer
    char * address1 = argv[2];  // no check, this is the address of the producer
    char * address2 = argv[3];  // no check, this is the address of the consumer

    // creates the network using 0mq as transport layer
    zmqTransport transport(atoi(P));
    if (transport.initTransport()<0) abort();

    if (atoi(P)) {
	ff_pipeline pipe;
	Node0* n0 = new Node0("B", address2, &transport);
	Node1* n1 = new Node1("A", address1, &transport);
	n0->skipfirstpop(true);
	pipe.add_stage(n0);
	pipe.add_stage(n1);	
	pipe.run_and_wait_end();
	delete n0;
	delete n1;
    } else {
	ff_pipeline pipe;
	Node2 * n2 = new Node2("A", address1, &transport);
	Node3 * n3 = new Node3("B", address2, &transport);
	pipe.add_stage(n2);
	pipe.add_stage(n3);	
	pipe.run_and_wait_end();
	delete n2;
	delete n3;
    }
    
    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
