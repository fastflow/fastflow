/*
 *
 *   -------------------------------------------------------------------
 *  |                                                 host0             |
 *  |                                          ----------------------   |  
 *  |                                         |                      |  |
 *  |                                         |                      |  |
 *  |          master                   ------|-> Worker --> Sender -|--
 *  |    --------------------------    |      |                      |
 *  |   |                          |   |      |    (ff_pipeline)     |          
 *  v   |                          |   |       ----------------------
 *   ---|-> Collector --> Emitter -|---            C           P 
 *  ^   |                          |   | ON-DEMAND (COMM1)
 *  |   |      (ff_pipeline)       |   | 
 *  |    --------------------------    |              host1
 *  |          C            P          |       ----------------------
 *  | FROM ANY (COMM2)                 |      |                      |
 *  |                                  |      |                      |
 *  |                                   ------|-> Worker --> Sender -|--
 *  |                                         |                      |  |
 *  |                                         |    (ff_pipeline)     |  |
 *  |                                          ----------------------   |
 *  |                                               C          P        |
 *   -------------------------------------------------------------------    
 *
 *   COMM1, the server address is master:port1 (the address1 parameter)
 *   COMM2, the server address is master:port2 (the address2 parameter)
 *
 */

#ifdef __linux
#include <sys/uio.h>
#endif
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdint.h>
#include <math.h>

// On win dnode.h should be included as first header to ensure winsock2.h comes before windows.h
// ZMQ requires winsock2.h and conflicts with windows.h
#include <ff/dnode.hpp>
#include <ff/node.hpp>
#include <ff/svector.hpp>
#include <ff/pipeline.hpp>
#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>

using namespace ff;

#define COMM1   zmqOnDemand
#define COMM2   zmqFromAny   
// gloabals just to save some coding
unsigned taskSize=0;
// the following SPSC unbounded queue is shared between the Worker and the Sender threads
// to free unused memory
uSWSR_Ptr_Buffer RECYCLE(1024); // NOTE: See also the farm_farm test for another solution to the same problem !


class Collector: public ff_dnode<COMM2> {
    typedef COMM2::TransportImpl        transport_t;
public:    
    Collector(unsigned nTasks, unsigned nHosts, const std::string& name, const std::string& address, transport_t* const transp):
	nTasks(nTasks),nHosts(nHosts),name(name),address(address),transp(transp) {
    }
    
    // initializes dnode
    int svc_init() {
	ff_dnode<COMM2>::init(name, address, nHosts, transp, RECEIVER,0);
	return 0;
    }
   
    void *svc(void *task) {
	if (task == NULL) {  
	    srandom(0); //::getpid()+(getusec()%4999));
	    for(unsigned i=0;i<nTasks;++i) {
		double* t = new double[taskSize*taskSize];
		assert(t);
		for(unsigned j=0;j<(taskSize*taskSize);++j)
		    t[j] = 1.0*random() / (double)(taskSize);
		
		ff_send_out((void*)t);
	    }
	    return GO_ON;
	}

	printf("got result\n");
	return GO_ON;
    }
private:
    unsigned nTasks;
    unsigned nHosts;
protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};


class Emitter: public ff_dnode<COMM1> {
    typedef COMM1::TransportImpl        transport_t;
protected:
    static void callback(void * e,void*) {
	delete [] (double*)e;
    }
public:
    Emitter(unsigned nTasks, unsigned nHosts, const std::string& name, const std::string& address, transport_t* const transp):
	nTasks(nTasks),nHosts(nHosts),name(name),address(address),transp(transp) {
    }
    
    int svc_init() {
	// the callback will be called as soon as the output message is no 
	// longer in use by the transport layer
	ff_dnode<COMM1>::init(name, address, nHosts, transp, SENDER, 0, callback);  
	return 0;
    }
    
    void * svc(void* task) {
	if (--nTasks == 0) {
	    ff_send_out(task);
	    return EOS; // generates EOS
	}
	return task;
    }

    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        struct iovec iov={ptr,taskSize*taskSize*sizeof(double)};
        v.push_back(iov);
    }    
private:
    unsigned nTasks;
    unsigned nHosts;
protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};

class Worker: public ff_dnode<COMM1> {
    typedef COMM1::TransportImpl        transport_t;
public:
    Worker(const std::string& name, const std::string& address, transport_t* const transp):
	name(name),address(address),transp(transp) {
    }
    
    int svc_init() {
	myt = new double[taskSize*taskSize];
	assert(myt);
	c = new double[taskSize*taskSize];
	assert(c);
	for(unsigned j=0;j<(taskSize*taskSize);++j)
	    c[j] = j*1.0;	

	ff_dnode<COMM1>::init(name, address, 1, transp, RECEIVER, transp->getProcId());  
	return 0;
    }

    void * svc(void * task) {
	static unsigned c=0;
	double* t = (double*)task;
	printf("Worker: get one task %d\n",++c);

	memset(myt,0,taskSize*taskSize*sizeof(double));
	
	for(unsigned i=0;i<taskSize;++i)
	    for(unsigned j=0;j<taskSize;++j)
		for(unsigned k=0;k<taskSize;++k)
		    myt[i*taskSize+j] += t[i*taskSize+k]*t[k*taskSize+j];
	
	memcpy(t,myt,taskSize*taskSize*sizeof(double));
	return t;
    }
    void svc_end() {
	delete [] myt;
	delete [] c;
    }

    void prepare(svector<msg_t*>*& v, size_t len, const int sender=-1) {
        svector<msg_t*> * v2 = new svector<msg_t*>(len);
        assert(v2);
        for(size_t i=0;i<len;++i) {
            msg_t * m = new msg_t;
            assert(m);
            v2->push_back(m);
	    RECYCLE.push(m);
        }
        v = v2;
    }

private:
    double * myt;
    double * c;
protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};

class Sender: public ff_dnode<COMM2> {
    typedef COMM2::TransportImpl        transport_t;
protected:
    static void callback(void *e,void*) {
        msg_t* p;
	if (!RECYCLE.pop((void**)&p)) abort();	
	delete p;
    }
public:
    Sender(const std::string& name, const std::string& address, transport_t* const transp):
	name(name),address(address),transp(transp) {
    }

    int svc_init() {
	// the callback will be called as soon as the output message is no 
	// longer in use by the transport layer
	ff_dnode<COMM2>::init(name, address, 1, transp, SENDER, transp->getProcId(),callback);  
	return 0;
    }

    void* svc(void* task) {
	printf("Sender, sending the task\n");
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


int main(int argc, char * argv[]) {    
    if (argc != 7) {
	std::cerr << "use: " << argv[0] << " tasksize streamlen 1|0 nhosts host:port1 host:port2\n";
	std::cerr << "  1 for the master 0 for other hosts\n";
	std::cerr << "  nhosts: is the number of hosts for the master and the hostID for the others\n";
	return -1;
    }
    
    taskSize = atoi(argv[1]);
    unsigned numTasks=atoi(argv[2]);
    char*    P = argv[3];                 // 1 for the master  0 for other hosts
    unsigned nhosts = atoi(argv[4]);      
    char*    address1  = argv[5];         // no check
    char*    address2  = argv[6];         // no check
    

    // creates the network using 0mq as transport layer
    zmqTransport transport(atoi(P)?-1 : nhosts);
    if (transport.initTransport()<0) abort();

    if (atoi(P)) {
	ff_pipeline pipe;
	Collector C(numTasks, nhosts, "B", address2, &transport);
	Emitter   E(numTasks, nhosts, "A", address1, &transport);

	C.skipfirstpop(true); 

	pipe.add_stage(&C);
	pipe.add_stage(&E);

	if (pipe.run_and_wait_end()<0) {
	    error("running pipeline\n");
	    return -1;
	}
	printf("Time= %f\n", pipe.ffwTime());
    } else {
	ff_pipeline pipe;
	Worker W("A", address1, &transport);
	Sender S("B", address2, &transport);
	pipe.add_stage(&W);
	pipe.add_stage(&S);

	if (!RECYCLE.init()) abort();

	if (pipe.run_and_wait_end()<0) {
	    error("running pipeline\n");
	    return -1;
	}
    }

    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
