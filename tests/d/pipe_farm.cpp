/*
 *
 *          host0                           host1
 *    ------------------             -------------------
 *   |                  |           |                   |
 *   |                  |  UNICAST  |                   |
 *   |     farm(f)  ----|---------- |---->  farm(g)     |
 *   |                  |           |                   |
 *    ------------------             -------------------
 *                 pipeline(farm(f),farm(g))
 *
 *         host0         
 *    ------------------ 
 *   |                  |
 *   |                  |
 *   |    farm(f;g)     |
 *   |                  |
 *    ------------------ 
 *
 *  If we define SINGLE at compile-time, then the code executed is on
 *  a single host as  farm(f;g).
 *
 *
 */

#include <sys/uio.h>
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdint.h>
#include <math.h>

#include <ff/node.hpp>
#include <ff/svector.hpp>
#include <ff/dnode.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>

using namespace ff;
#define COMM   zmq1_1

// gloabal just to save some coding
unsigned taskSize=0;

class Emitter: public ff_node {
public:
    Emitter(unsigned nTasks):nTasks(nTasks) {}
    void * svc(void*) {
	srandom(0); //::getpid()+(getusec()%4999));

	for(unsigned i=0;i<nTasks;++i) {
	  double* t = new double[taskSize*taskSize];
	  assert(t);
	  for(unsigned j=0;j<(taskSize*taskSize);++j)
	    t[j] = 1.0*random() / (double)(taskSize);
	  
	  ff_send_out((void*)t);
	}
	return NULL;
    }
private:
    unsigned nTasks;
};

class Worker1: public ff_node {
public:
    
    int svc_init() {
	myt = new double[taskSize*taskSize];
	assert(myt);
	c = new double[taskSize*taskSize];
	assert(c);
	for(unsigned j=0;j<(taskSize*taskSize);++j)
	    c[j] = j*1.0;	
	return 0;
    }

    void * svc(void * task) {
	// f
	double* t = (double*)task;
	bzero(myt,taskSize*taskSize*sizeof(double));
	
	for(unsigned i=0;i<taskSize;++i)
	    for(unsigned j=0;j<taskSize;++j)
		for(unsigned k=0;k<taskSize;++k)
		    myt[i*taskSize+j] += t[i*taskSize+k]*t[k*taskSize+j];
	
	memcpy(t,myt,taskSize*taskSize*sizeof(double));
#if defined(SINGLE)
	// g
	bzero(myt,taskSize*taskSize*sizeof(double));
	for(unsigned i=0;i<taskSize;++i)
	    for(unsigned j=0;j<taskSize;++j)
		for(unsigned k=0;k<taskSize;++k)
		    myt[i*taskSize+j] += t[i*taskSize+k]*c[k*taskSize+j];
	
	memcpy(t,myt,taskSize*taskSize*sizeof(double));
#endif
	return t;
    }
    void svc_end() {
	delete [] myt;
	delete [] c;
    }
private:
    double * myt;
    double * c;
};


class Collector: public ff_dnode<COMM> {
protected:
    static void callback(void * e,void*) {
	delete [] (double*)e;
    }
public:
    typedef COMM::TransportImpl        transport_t;
    
    Collector(const std::string& name, const std::string& address, transport_t* const transp):
	name(name),address(address),transp(transp) {
    }
    
#if !defined(SINGLE)
    // initializes dnode
    int svc_init() {
	// the callback will be called as soon as the output message is no 
	// longer in use by the transport layer
	ff_dnode<COMM>::init(name, address, 1, transp, SENDER, 0, callback);  
	return 0;
    }
#endif
    
    void *svc(void *task) {
#if defined(SINGLE)
	callback(task,NULL);
	return GO_ON;
#endif
	return task;
    }
    
    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        struct iovec iov={ptr,taskSize*taskSize*sizeof(double)};
        v.push_back(iov);
    }    
    
protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};


class Emitter2: public ff_dnode<COMM> {
    typedef COMM::TransportImpl        transport_t;
public:
    Emitter2(const std::string& name, const std::string& address, transport_t* const transp):
	name(name),address(address),transp(transp) {
    }

    // initializes dnode
    int svc_init() {
	ff_dnode<COMM>::init(name, address, 1, transp, RECEIVER);  
	return 0;
    }

    void * svc(void* task) {
	printf("Emitter2 received one task\n");
	assert(task);
	return task;
    }
protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};


class Worker2: public ff_node {
public:

  int svc_init() {
    myt = new double[taskSize*taskSize];
    assert(myt);
    c = new double[taskSize*taskSize];
    assert(c);
    for(unsigned j=0;j<(taskSize*taskSize);++j)
      c[j] = j*1.0;
	  
    return 0;
  }
    void * svc(void * task) {
      double* t = (double*)task;
      bzero(myt,taskSize*taskSize*sizeof(double));
      for(unsigned i=0;i<taskSize;++i)
        for(unsigned j=0;j<taskSize;++j)
          for(unsigned k=0;k<taskSize;++k)
	    myt[i*taskSize+j] += t[i*taskSize+k]*c[k*taskSize+j];

      memcpy(t,myt,taskSize*taskSize*sizeof(double));
      return t;
    }
  void svc_end() {
    delete [] myt;
    delete [] c;
  }
private:
  double * myt;
  double * c;
};

class Collector2: public ff_node {
public:
    void *svc(void* task) {
	printf("got task\n");
	return GO_ON;
    }
};




int main(int argc, char * argv[]) {    
    if (argc != 7) {
	std::cerr << "use: " << argv[0] << " nw tasksize streamlen name 1|0 host:port\n";
	return -1;
    }
    
    unsigned nw = atoi(argv[1]);
    taskSize = atoi(argv[2]);
    unsigned numTasks=atoi(argv[3]);
    char * name = argv[4];
    char * P = argv[5];        // 1 producer 0 consumer
    char * address = argv[6];  // no check


    // creates the network using 0mq as transport layer
    zmqTransport transport(atoi(P)?0:1);
    if (transport.initTransport()<0) abort();

    if (atoi(P)) {
	ff_farm<> farm;
	Emitter E(numTasks);
	farm.add_emitter(&E);
	Collector C(name, address, &transport);
	farm.add_collector(&C);
	std::vector<ff_node *> w;
	for(unsigned i=0;i<nw;++i) w.push_back(new Worker1);
	farm.add_workers(w); // add all workers to the farm

	if (farm.run_and_wait_end()<0) {
	    error("running farm\n");
	    return -1;
	}
	printf("Time= %f\n", farm.ffwTime());

#if defined(TRACE_FASTFLOW)
	farm.ffStats(std::cout);
#endif
    } else {

	ff_farm<> farm2;
	Emitter2 E2(name, address, &transport);
	farm2.add_emitter(&E2);
	Collector2 C2;
	farm2.add_collector(&C2);
	std::vector<ff_node *> w2;
	for(unsigned i=0;i<nw;++i) w2.push_back(new Worker2);
	farm2.add_workers(w2);

	if (farm2.run_and_wait_end()<0) {
	    error("running farm\n");
	    return -1;
	}
	printf("Time= %f\n", farm2.ffwTime());
    }

    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
