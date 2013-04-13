/*
 * Picture:
 *
 *                                                  Farm0        
 *                                           ------------------     
 *                                          |                  |  
 *                                          |                  |  
 *             First                  ------|-----> Farm ------|---                      Last
 *     --------------------------    |      |                  |   |            ---------------------------
 *    |                          |   |      |     (ff_farm)    |   |           |                           |
 *    |                          |   | A     ------------------    | B         |                           |  C
 *  --|-> StartStop --> Emitter--|---                              ------------| --> Collector --> Sender--|---
 * |  |                          |   | ON-DEMAND (COMM1)           |           |                           |   |
 * |  |      (ff_pipeline)       |   |                             |           |                           |   |
 * |   --------------------------    |              Farm1          |           |       (ff_pipeline)       |   |
 * |                                 |       ------------------    |            ---------------------------    |
 * |                                 |      |                  |   |                                           |
 * |                                 |      |                  |   |                                           |
 * |                                  ------|------> Farm -----|---                                            |
 * |                                        |                  |   FROM ANY (COMM2)                            |
 * |                                        |      (ff_farm)   |                                               |
 * |                                         ------------------                                                |
 * |                                                                                  UNICAST (COMM3)          |
 *  -----------------------------------------------------------------------------------------------------------
 *
 *  NOTE: - Each Farm has the same number of workers (nw)
 *        
 *        
 *
 *   COMM1, the server address is master:port1 (the address1 parameter)
 *   COMM2, the server address is master:port2 (the address2 parameter)
 *   COMM3, the server address is master:port3 (the address3 parameter)
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
#if defined(USE_PROC_AFFINITY)
#include <ff/mapping_utils.hpp>
#endif
#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>

using namespace ff;

#define COMM1   zmqOnDemand
#define COMM2   zmqFromAny   
#define COMM3   zmq1_1

// gloabals just to save some coding
unsigned taskSize=0;

#if defined(USE_PROC_AFFINITY)
//WARNING: the following mapping targets dual-eight core Intel Sandy-Bridge 
const int worker_mapping[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
const int emitter_mapping  = 31;
const int PHYCORES         = 16;
#endif


struct ff_task_t { 
    double* getData() { return (double*)(msg->getData()); }	
    zmqTransport::msg_t*   msg;
    ff_task_t*  self;
};


/* -------------- First ------------------- */

class StartStop: public ff_dnode<COMM3> {
    typedef COMM2::TransportImpl        transport_t;
public:    
    StartStop(unsigned nTasks, const std::string& name, const std::string& address, transport_t* const transp):
	nTasks(nTasks),name(name),address(address),transp(transp) {
    }
    
    // initializes dnode
    int svc_init() {
	struct timeval start,stop;
        gettimeofday(&start,NULL);
	printf("StartStop init Id= %d\n", transp->getProcId());
	ff_dnode<COMM3>::init(name, address, 1, transp, RECEIVER,transp->getProcId());
        gettimeofday(&stop,NULL);
	
	printf("StartStop init time %f ms\n", diffmsec(stop,start));

	cnt=0;
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

	// just free memory for each input task
	ff_task_t* t = (ff_task_t*)task;
	delete (t->msg);
	delete (t->self);

	++cnt;
	return GO_ON;
    }

    void svc_end() {
	printf("StartStop: computed %d tasks\n", cnt);
    }

  void prepare(svector<msg_t*>*& v, size_t len, const int sender=-1) {
        svector<msg_t*> * v2 = new svector<msg_t*>(len);
        assert(v2);
        for(size_t i=0;i<len;++i) {
            msg_t * m = new msg_t;
            assert(m);
            v2->push_back(m);
        }
        v = v2;
    }

    virtual void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
        assert(vlen==1 && v[0]->size()==1); 
	ff_task_t* t = new ff_task_t;
	t->msg = v[0]->operator[](0);
	t->self= t;
	task   = t;
        delete v[0];
    }

private:
    unsigned nTasks;
protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
    unsigned cnt;
};


class Emitter: public ff_dnode<COMM1> {
    typedef COMM1::TransportImpl        transport_t;
protected:
    static void callback(void* e, void* arg) {
	delete [] (double*)e;
    }
public:
    Emitter(unsigned nTasks, unsigned nHosts, const std::string& name, const std::string& address, transport_t* const transp):
	nTasks(nTasks),nHosts(nHosts),name(name),address(address),transp(transp) {
    }
    
    int svc_init() {
	struct timeval start,stop;
        gettimeofday(&start,NULL);
	printf("Emitter init Id= %d\n", transp->getProcId());
	ff_dnode<COMM1>::init(name, address, nHosts, transp, SENDER, transp->getProcId(), callback);  

        gettimeofday(&stop,NULL);
	printf("Emitter init time %f ms\n", diffmsec(stop,start));
	return 0;
    }
    
    void * svc(void* task) {
	if (--nTasks == 0) {
	    ff_send_out(task);
	    return NULL; // generates EOS
	}
	return task;
    }

    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        struct iovec iov={ptr,taskSize*taskSize*sizeof(double)};
        v.push_back(iov);
	setCallbackArg(NULL);
    }    
private:
    unsigned nTasks;
    unsigned nHosts;
protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};


/* -------------- Farm ------------------- */

class Emitter2: public ff_dnode<COMM1> {
    typedef COMM1::TransportImpl        transport_t;
public:
    Emitter2(const std::string& name, const std::string& address, transport_t* const transp):
	name(name),address(address),transp(transp) {
    }
    
    int svc_init() {
	printf("Emitter2 init Id= %d\n", transp->getProcId());
	ff_dnode<COMM1>::init(name, address, 1, transp, RECEIVER, transp->getProcId());  
	return 0;
    }

    void * svc(void * task) {
	return task;
    }

    void prepare(svector<msg_t*>*& v, size_t len, const int sender=-1) {
        svector<msg_t*> * v2 = new svector<msg_t*>(len);
        assert(v2);
        for(size_t i=0;i<len;++i) {
            msg_t * m = new msg_t;
            assert(m);
            v2->push_back(m);
        }
        v = v2;
    }

    virtual void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
        assert(vlen==1 && v[0]->size()==1); 
	ff_task_t* t = new ff_task_t;
	t->msg = v[0]->operator[](0);
	t->self= t;
	task   = t;
        delete v[0];
    }

protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};

class Worker: public ff_node {
public:
    int svc_init() {
	myt = new double[taskSize*taskSize];
	assert(myt);
	cnt=0;
	return 0;
    }
    
    void * svc(void * task) {
	double* t = ((ff_task_t*)task)->getData();
	bzero(myt,taskSize*taskSize*sizeof(double));
	
	for(unsigned i=0;i<taskSize;++i)
	    for(unsigned j=0;j<taskSize;++j)
		for(unsigned k=0;k<taskSize;++k)
#if defined(OPTIMIZE_CACHE)
		    myt[j*taskSize+k] += t[j*taskSize+i]*t[i*taskSize+k];
#else
		    myt[i*taskSize+j] += t[i*taskSize+k]*t[k*taskSize+j];
#endif
	
	memcpy(t,myt,taskSize*taskSize*sizeof(double));
	++cnt;
	return task;
    }
    void svc_end() {
	delete [] myt;
	printf("Worker: computed %d tasks\n", cnt);
    }

private:
    double * myt;
    unsigned cnt;
};


class Collector2: public ff_dnode<COMM2> {
   typedef COMM2::TransportImpl        transport_t;
protected:
    static void callback(void *e, void* arg) {
	ff_task_t* t = (ff_task_t*)arg;
	assert(t);
	delete (t->msg);
	delete (t->self);
    }
public:
    Collector2(const std::string& name, const std::string& address, transport_t* const transp):
	name(name),address(address),transp(transp) {
    }

    int svc_init() {
	printf("Collector2 init Id= %d\n", transp->getProcId());
	ff_dnode<COMM2>::init(name, address, 1, transp, SENDER, transp->getProcId());  
	return 0;
    }
    
    void * svc(void * task) {
	return task;
    }

    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        struct iovec iov={((ff_task_t*)ptr)->getData(),taskSize*taskSize*sizeof(double)};
        v.push_back(iov);
	setCallbackArg(ptr);
    }    

protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};


/* -------------- Last ------------------- */

class Collector: public ff_dnode<COMM2> {
    typedef COMM2::TransportImpl        transport_t;
public:

    Collector(const std::string& name, unsigned nHosts, const std::string& address, transport_t* const transp):
	name(name),nHosts(nHosts), address(address),transp(transp) {
    }
    
    int svc_init() {
	printf("Collector init Id= %d\n", transp->getProcId());
	ff_dnode<COMM2>::init(name, address, nHosts, transp, RECEIVER, transp->getProcId());  
	return 0;
    }

    void * svc(void * task) {
	return task;
    }

    void prepare(svector<msg_t*>*& v, size_t len, const int sender=-1) {
        svector<msg_t*> * v2 = new svector<msg_t*>(len);
        assert(v2);
        for(size_t i=0;i<len;++i) {
            msg_t * m = new msg_t;
            assert(m);
            v2->push_back(m);
        }
        v = v2;
    }

    virtual void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
        assert(vlen==1 && v[0]->size()==1); 
	ff_task_t* t = new ff_task_t;
	t->msg = v[0]->operator[](0);
	t->self= t;
	task   = t;
        delete v[0];
    }

protected:
    const std::string name;
    const unsigned    nHosts;
    const std::string address;
    transport_t   * transp;
};


class Sender: public ff_dnode<COMM3> {
    typedef COMM3::TransportImpl        transport_t;
protected:
    static void callback(void *e, void* arg) {
	ff_task_t* t = (ff_task_t*)arg;
	assert(t);
	delete (t->msg);
	delete (t->self);
    }
public:
    Sender(const std::string& name, const std::string& address, transport_t* const transp):
	name(name),address(address),transp(transp) {
    }

    int svc_init() {
	printf("Sender init Id= %d\n", transp->getProcId());
	ff_dnode<COMM3>::init(name, address, 1, transp, SENDER, transp->getProcId());  
	return 0;
    }
    
    void * svc(void * task) {
	return task;
    }

    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        struct iovec iov={((ff_task_t*)ptr)->getData(),taskSize*taskSize*sizeof(double)};
        v.push_back(iov);
	setCallbackArg(ptr);
    }    

protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};


/* --------------------------------- */

int main(int argc, char * argv[]) {    

    if (argc == 2 && (atoi(argv[1])<0)) {
	printf(
 "                                                  Farm0                                                        \n"
 "                                           ------------------                                                  \n"
 "                                          |                  |                                                 \n"
 "                                          |                  |                                                 \n"
 "             First                  ------|-----> Farm ------|---                      Last                    \n"
 "     --------------------------    |      |                  |   |            ---------------------------      \n"
 "    |                          |   |      |     (ff_farm)    |   |           |                           |     \n"
 "    |                          |   | A     ------------------    | B         |                           |  C  \n"
 "  --|-> StartStop --> Emitter--|---                              ------------| --> Collector --> Sender--|---  \n"
 " |  |                          |   | ON-DEMAND (COMM1)           |           |                           |   | \n"
 " |  |      (ff_pipeline)       |   |                             |           |                           |   | \n"
 " |   --------------------------    |              Farm1          |           |       (ff_pipeline)       |   | \n"
 " |                                 |       ------------------    |            ---------------------------    | \n"
 " |                                 |      |                  |   |                                           | \n"
 " |                                 |      |                  |   |                                           | \n"
 " |                                  ------|------> Farm -----|---                                            | \n"
 " |                                        |                  |   FROM ANY (COMM2)                            | \n"
 " |                                        |      (ff_farm)   |                                               | \n"
 " |                                         ------------------                                                | \n"
 " |                                                                                  UNICAST (COMM3)          | \n"
 "  -----------------------------------------------------------------------------------------------------------  \n"
 "                                                                                                               \n"
 "   COMM1, the server address is master:port1 (the address1 parameter)                                          \n"
 "   COMM2, the server address is master:port2 (the address2 parameter)                                          \n"
 "   COMM3, the server address is master:port3 (the address3 parameter)                                          \n");
	return 0;
    }


    if (argc < 9) {
	std::cerr << "\n";
	std::cerr << "use: " << argv[0] << " tasksize stream-length hostId nfarms nw host:port1 host:port2 host:port3\n";
	std::cerr << "  hostId: the host identifier, for the farms the hostId(s) are in the range [0..N[\n";
	std::cerr << "          -1 is the id of the first stage (First in the picture)\n";
	std::cerr << "          -2 is the id of the last stage (Last in the picture)\n"; 
	std::cerr << "  nfarms: number of farm hosts\n";
	std::cerr << "  nw: number of farm's worker\n\n";
	std::cerr << " To print the application picture use " << argv[0] << " -1\n\n\n";
	return -1;
    }
    
    // SPMD style code

    taskSize = atoi(argv[1]);

    unsigned numTasks=atoi(argv[2]);
    int      hostId = atoi(argv[3]);    
    unsigned nfarms = atoi(argv[4]);      
    unsigned nw     = atoi(argv[5]);
    char*    address1  = argv[6];         // no check
    char*    address2  = argv[7];         // no check
    char*    address3  = argv[8];         // no check
    

    // creates the network using 0mq as transport layer
    zmqTransport transport(hostId);
    if (transport.initTransport()<0) abort();

    if (hostId == -1) {   // First
	ff_pipeline pipe;
	StartStop C(numTasks,"C", address3, &transport);
	Emitter   E(numTasks, nfarms, "A", address1, &transport);

	C.skipfirstpop(true); 

	pipe.add_stage(&C);
	pipe.add_stage(&E);

	if (pipe.run_and_wait_end()<0) {
	    error("running pipeline\n");
	    return -1;
	}
	printf("wTime= %f\n", pipe.ffwTime());
	printf("Time = %f\n", pipe.ffTime());
    } else if (hostId == -2) {  // Last
	ff_pipeline pipe;
	Collector C("B", nfarms, address2, &transport);
	Sender    E("C", address3, &transport);

	pipe.add_stage(&C);
	pipe.add_stage(&E);

	if (pipe.run_and_wait_end()<0) {
	    error("running pipeline\n");
	    return -1;
	}
    } else {  // Farm(s)
	ff_farm<> farm;
	Emitter2 E2("A", address1, &transport);
	farm.add_emitter(&E2);
	std::vector<ff_node *> w;
	for(unsigned i=0;i<nw;++i) w.push_back(new Worker);
	farm.add_workers(w); // add all workers to the farm
	Collector2 C2("B",address2, &transport);
	farm.add_collector(&C2);

	if (farm.run_and_wait_end()<0) {
	    error("running farm\n");
	    return -1;
	}
    }

    transport.closeTransport();

    std::cout << "done\n";
    return 0;
}
