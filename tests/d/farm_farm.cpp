/*
 *
 *   ---------------------------------------------------------------
 *  |                                                 host0         |
 *  |                                          ------------------   |  
 *  |                                         |                  |  |
 *  |                                         |                  |  |
 *  |          master                   ------|-----> Farm ------|--
 *  |    --------------------------    |      |                  |
 *  |   |                          |   |      |     (ff_farm)    |          
 *  v   |                          |   |       ------------------
 *   ---|-> Collector --> Emitter--|---          
 *  ^   |                          |   | ON-DEMAND (COMM1)
 *  |   |      (ff_pipeline)       |   | 
 *  |    --------------------------    |              host1
 *  |          C            P          |       ------------------
 *  | FROM ANY (COMM2)                 |      |                  |
 *  |                                  |      |                  |
 *  |                                   ------|------> Farm -----|--
 *  |                                         |                  |  |
 *  |                                         |      (ff_farm)   |  |
 *  |                                          ------------------   |
 *  |                                                               |
 *   ---------------------------------------------------------------    
 *  NOTE: - Each Farm has the same number of workers (nw)
 *        - The Farm does not have the collector thus each worker sends
 *          data to the Collector.
 *
 *   COMM1, the server address is master:port1 (the address1 parameter)
 *   COMM2, the server address is master:port2 (the address2 parameter)
 *
 * NOTE:
 * If SINGLE_FARM is defined at compile time, then a single master-worker 
 * farm skeleton is used to compute all the matrices.
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
// gloabals just to save some coding
unsigned taskSize=0;

#if defined(USE_PROC_AFFINITY)
//WARNING: the following mapping targets dual-eight core Intel Sandy-Bridge 
const int worker_mapping[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
const int emitter_mapping  = 31;
const int PHYCORES         = 16;
#endif


class Collector: public ff_dnode<COMM2> {
    typedef COMM2::TransportImpl        transport_t;
public:    
    Collector(unsigned nTasks, unsigned nHosts, const std::string& name, const std::string& address, transport_t* const transp):
	nTasks(nTasks),nHosts(nHosts),name(name),address(address),transp(transp) {
    }
    
    // initializes dnode
    int svc_init() {
#if !defined(SINGLE_FARM)
      struct timeval start,stop;
        gettimeofday(&start,NULL);
	ff_dnode<COMM2>::init(name, address, nHosts, transp, RECEIVER,0);
        gettimeofday(&stop,NULL);

	printf("Collector init time %f ms\n", diffmsec(stop,start));
#endif

#if defined(USE_PROC_AFFINITY)
	if (ff_mapThreadToCpu(emitter_mapping)!=0)
	  printf("Cannot map Emitter to CPU %d\n",emitter_mapping);
	//else  printf("Emitter mapped to CPU %d\n", emitter_mapping);
#endif

	cnt=0;
	return 0;
    }
#if !defined(SINGLE_FARM)   
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

	//printf("got result\n");
	++cnt;
	return GO_ON;
    }
#else
    void *svc(void *task) {
	static unsigned sent=0;
	if (task == NULL) {  
	  const unsigned TASK_BATCH=(nTasks<1024)?nTasks:1024;

	    srandom(0); //::getpid()+(getusec()%4999));
	    for(unsigned i=0;i<TASK_BATCH;++i) {
		double* t = new double[taskSize*taskSize];
		assert(t);
		for(unsigned j=0;j<(taskSize*taskSize);++j)
		    t[j] = 1.0*random() / (double)(taskSize);
		
		ff_send_out((void*)t);
	    }
	    sent+=TASK_BATCH;
	    return GO_ON;
	}

	//printf("got result\n");
	++cnt;
	if (cnt==nTasks) {
	    delete [] (double*)task;
	    return NULL; // generates EOS
	}
	if (sent<nTasks) {
	    double* t= (double*)task;
	    for(unsigned j=0;j<(taskSize*taskSize);++j)
		t[j] = 1.0*random() / (double)(taskSize);	    
	    ff_send_out((void*)t);
	    ++sent;
	} else delete [] (double*)task;

	return GO_ON;
    }
#endif

  void svc_end() {
    printf("Collector: computed %d tasks\n", cnt);
  }

private:
    unsigned nTasks;
    unsigned nHosts;
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
	// the callback will be called as soon as the output message is no 
	// longer in use by the transport layer
      struct timeval start,stop;
        gettimeofday(&start,NULL);
	ff_dnode<COMM1>::init(name, address, nHosts, transp, SENDER, 0, callback);  

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


struct ff_task_t { 
    double* getData() { return (double*)(msg->getData()); }	
    COMM1::TransportImpl::msg_t*   msg;
    ff_task_t*  self;
};


class Emitter2: public ff_dnode<COMM1> {
    typedef COMM1::TransportImpl        transport_t;
public:
    Emitter2(const std::string& name, const std::string& address, transport_t* const transp):
	name(name),address(address),transp(transp) {
    }
    
    int svc_init() {
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
	task = t;
        //task = v[0]->operator[](0)->getData();
        delete v[0];
    }

protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};

class Worker: public ff_dnode<COMM2> {
    typedef COMM2::TransportImpl        transport_t;
protected:
    static void callback(void *e, void* arg) {
	ff_task_t* t = (ff_task_t*)arg;
	assert(t);
	delete (t->msg);
	delete (t->self);
    }
public:
    Worker(const int nw, const std::string& name, const std::string& address, transport_t* const transp):
	nw(nw),name(name),address(address),transp(transp) {
    }

    int svc_init() {
	myt = new double[taskSize*taskSize];
	assert(myt);
	c = new double[taskSize*taskSize];
	assert(c);
	for(unsigned j=0;j<(taskSize*taskSize);++j)
	    c[j] = j*1.0;	

#if !defined(SINGLE_FARM)
	ff_dnode<COMM2>::init(name, address, 1, transp, SENDER, transp->getProcId()*nw+get_my_id());  
#endif

#if defined(USE_PROC_AFFINITY)
	if (ff_mapThreadToCpu(worker_mapping[get_my_id() % PHYCORES])!=0)
	  printf("Cannot map Worker %d CPU %d\n",get_my_id(),
		 worker_mapping[get_my_id() % PHYCORES]);
	//else printf("Thread %d mapped to CPU %d\n",get_my_id(), worker_mapping[get_my_id() % PHYCORES]);
#endif

	cnt=0;
	return 0;
    }
    
    void * svc(void * task) {
#if !defined(SINGLE_FARM)
	double* t = ((ff_task_t*)task)->getData();
#else
	double* t = (double*)task;
#endif
	bzero(myt,taskSize*taskSize*sizeof(double));
	
	for(unsigned i=0;i<taskSize;++i)
	    for(unsigned j=0;j<taskSize;++j)
		for(unsigned k=0;k<taskSize;++k)
		    myt[i*taskSize+j] += t[i*taskSize+k]*t[k*taskSize+j];
	
	memcpy(t,myt,taskSize*taskSize*sizeof(double));
	++cnt;
	return task;
    }
    void svc_end() {
	delete [] myt;
	delete [] c;

	printf("Worker: computed %d tasks\n", cnt);
    }

    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        struct iovec iov={((ff_task_t*)ptr)->getData(),taskSize*taskSize*sizeof(double)};
        v.push_back(iov);
	setCallbackArg(ptr);
    }    

private:
    double * myt;
    double * c;
protected:
    const int nw;
    const std::string name;
    const std::string address;
    transport_t   * transp;
  
  unsigned cnt;
};



int main(int argc, char * argv[]) {    
    if (argc != 8) {
	std::cerr << "use: " << argv[0] << " tasksize streamlen 1|0 nhosts nw host:port1 host:port2\n";
	std::cerr << "  1 for the master 0 for other hosts\n";
	std::cerr << "  nhosts: is the number of hosts for the master and the hostID for the others\n";
	std::cerr << "  nw: number of farm's worker\n";
	return -1;
    }
    
    taskSize = atoi(argv[1]);
    unsigned numTasks=atoi(argv[2]);
    char*    P = argv[3];                 // 1 for the master  0 for other hosts
    unsigned nhosts = atoi(argv[4]);      
    unsigned nw     = atoi(argv[5]);
    char*    address1  = argv[6];         // no check
    char*    address2  = argv[7];         // no check
    

#if defined(SINGLE_FARM) 
    ff_farm<> farm;
    Collector C(numTasks, nhosts, "A", address1, NULL);
    farm.add_emitter(&C);
    std::vector<ff_node *> w;
    for(unsigned i=0;i<nw;++i) w.push_back(new Worker(nw, "B",address2,NULL));
    farm.add_workers(w); // add all workers to the farm

    farm.wrap_around(); // setting master-worker computation

    if (farm.run_and_wait_end()<0) {
	error("running pipeline\n");
	return -1;
    }   
    printf("wTime= %f\n", farm.ffwTime());
    printf("Time = %f\n", farm.ffTime());
#else

    // creates the network using 0mq as transport layer
    zmqTransport transport(atoi(P)?-1 : nhosts);
    if (transport.initTransport()<0) abort();

    if (atoi(P)) {
	ff_pipeline pipe;
	Collector C(numTasks, nw*nhosts, "B", address2, &transport);
	Emitter   E(numTasks, nhosts, "A", address1, &transport);

	C.skipfirstpop(true); 

	pipe.add_stage(&C);
	pipe.add_stage(&E);

	if (pipe.run_and_wait_end()<0) {
	    error("running pipeline\n");
	    return -1;
	}
	printf("wTime= %f\n", pipe.ffwTime());
	printf("Time = %f\n", pipe.ffTime());
    } else {
	ff_farm<> farm;
	Emitter2 E2("A", address1, &transport);
	farm.add_emitter(&E2);
	std::vector<ff_node *> w;
	for(unsigned i=0;i<nw;++i) w.push_back(new Worker(nw, "B",address2,&transport));
	farm.add_workers(w); // add all workers to the farm

	if (farm.run_and_wait_end()<0) {
	    error("running farm\n");
	    return -1;
	}
    }

    transport.closeTransport();
#endif

    std::cout << "done\n";
    return 0;
}
