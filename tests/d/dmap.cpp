/*
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
 *  ^   |                          |   | SCATTER (COMM1)
 *  |   |      (ff_pipeline)       |   | 
 *  |    --------------------------    |              host1
 *  |          C            P          |       ----------------------
 *  | ALLGATHER (COMM2)                |      |                      |
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
 *  Distributed map example.
 *
 */

#include <map>
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

#define COMM1   zmqScatter
#define COMM2   zmqAllGather

// gloabals just to save some coding 
unsigned matSize=0; // this is the number of rows and columns of the matrix

struct ff_task_t {
    ff_task_t(double* t, unsigned numRows):
	numRows(numRows),task(t) {}
    unsigned numRows;
    double*  task;
};

// used to recycle memory
struct taskPtr_t {    
    ff_task_t* t;
    void* ptr;
};

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
	myt = new double[matSize*matSize]; // set maximum size
	ff_dnode<COMM2>::init(name, address, nHosts, transp, RECEIVER,0);
	ff::ffTime(START_TIME);
	return 0;
    }
    
    void *svc(void *task) {
	if (task == NULL) {  
	    srandom(0); //::getpid()+(getusec()%4999));
	    for(unsigned i=0;i<nTasks;++i) {
		double* t = new double[matSize*matSize];
		assert(t);
		for(unsigned j=0;j<(matSize*matSize);++j)
		    t[j] = 1.0*random() / (double)(matSize);
		
		ff_send_out((void*)t);
	    }
	    return GO_ON;
	}
	
	printf("got matrix result\n");
	return GO_ON;
    }

    void svc_end() {
	ff::ffTime(STOP_TIME);
	printf("Computation Time %.2f (ms)\n", ff::ffTime(GET_TIME));
    }

    double ffTime() { return ff_node::ffTime(); }

    virtual void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
	assert(vlen == (int)nHosts);
	for(int i=0;i<vlen;++i) {
	    assert(v[i]->size() == 2);
	    ff_task_t* t = static_cast<ff_task_t*>(v[i]->operator[](0)->getData());
	    t->task   = static_cast<double*>(v[0]->operator[](1)->getData());	    
	    memcpy(myt+(i*t->numRows),t->task, t->numRows*matSize*sizeof(double));
        }    
	task=myt;
    }
    
private:
    unsigned nTasks;
    unsigned nHosts;
    double * myt;
protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};


class Emitter: public ff_dnode<COMM1> {
    typedef COMM1::TransportImpl        transport_t;
protected:
    static void callback(void * e,void* arg) {
       	taskPtr_t* t = (taskPtr_t*)arg;
	if (!t) return;
	delete t->t;	
	if ( --M[t->ptr] == 0) {
	    delete [] (double*)t->ptr;	
	}
	delete t;
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
	    return NULL; // generates EOS
	}
	return task;
    }

    // Since COMM1 is a scatter pattern, the prepare method will be called 
    // nHosts times, once for each message to send.
    void prepare(svector<iovec>& v, void* ptr, const int sender) {
	assert(sender != -1);
	unsigned start=0, numRows=0;
	unsigned rowsize=matSize*sizeof(double);

	getPartition(sender,start,numRows);
	ff_task_t* t = new ff_task_t((double*)ptr+(start*matSize),numRows+1);

        struct iovec iov={t, sizeof(ff_task_t)};
	v.push_back(iov);
	setCallbackArg(NULL);

        iov.iov_base=t->task;
	iov.iov_len =t->numRows*rowsize;
        v.push_back(iov);
	taskPtr_t* tptr = new taskPtr_t;
	tptr->t=t;
	tptr->ptr=ptr;
	setCallbackArg(tptr);

	if (M.find(ptr) == M.end()) {
	    M[ptr]=nHosts;
	}
    }    
private:
    inline void getPartition(const int Id, unsigned& start, unsigned& size) {
        int r = matSize / nHosts;
        const int m = matSize % nHosts;
        const int st = (Id * r) + ((m >= Id) ? Id : m);
        if (Id < m) ++r;
	start = st; size= r-1;
    }
private:
    unsigned nTasks;
    unsigned nHosts;
    static std::map<void*, unsigned> M;

protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};
std::map<void*,unsigned> Emitter::M;


class Worker: public ff_dnode<COMM1> {
    typedef COMM1::TransportImpl        transport_t;
public:
    Worker(const std::string& name, const std::string& address, transport_t* const transp):
	name(name),address(address),transp(transp) {
    }
    
    int svc_init() {
	myt = new double[matSize*matSize]; // set maximum size
	assert(myt);

	ff_dnode<COMM1>::init(name, address, 1, transp, RECEIVER, transp->getProcId());  
	return 0;
    }

    void * svc(void * task) {
	ff_task_t* _t = (ff_task_t*)task;
	double* t  = _t->task;
	const unsigned numRows=_t->numRows;
	
	//printf("Worker: get one task, numRows=%d\n", numRows);
	memset(myt,0,matSize*matSize*sizeof(double));	
	for(unsigned i=0;i<numRows;++i)
	    for(unsigned j=0;j<matSize;++j)
	  	for(unsigned k=0;k<numRows;++k)
	  	    myt[i*matSize+j] += t[i*matSize+k]*t[k*matSize+j];
	
	memcpy(t,myt,numRows*matSize*sizeof(double));
        
	_t->task = t;
	return _t;
    }
    void svc_end() {
	delete [] myt;
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

    virtual void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
        assert(vlen==1 && v[0]->size()==2); 
	ff_task_t* t =static_cast<ff_task_t*>(v[0]->operator[](0)->getData());
        t->task   =static_cast<double*>(v[0]->operator[](1)->getData());
	task=t;
    }

private:
    double * myt;
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
	//printf("Sender, sending the task, numRows=%d\n", ((ff_task_t*)task)->numRows);
	return task;
    }

    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
	ff_task_t* t = (ff_task_t*)ptr;
        struct iovec iov={t, sizeof(ff_task_t)};
	v.push_back(iov);
	setCallbackArg(NULL);
        iov.iov_base= t->task;
	iov.iov_len = t->numRows*matSize*sizeof(double);
        v.push_back(iov);
	setCallbackArg(NULL);
    }    
protected:
    const std::string name;
    const std::string address;
    transport_t   * transp;
};


int main(int argc, char * argv[]) { 
    if (argc == 2 && (atoi(argv[1])<0)) {
	printf(
 "   -------------------------------------------------------------------   \n"
 "  |                                                 host0             |  \n"
 "  |                                          ----------------------   |  \n"
 "  |                                         |                      |  |  \n"
 "  |                                         |                      |  |  \n"
 "  |          host-master              ------|-> Worker --> Sender -|--   \n"
 "  |    --------------------------    |      |                      |     \n"
 "  |   |                          |   |      |    (ff_pipeline)     |     \n"     
 "  v   |                          |   |       ----------------------      \n"
 "   ---|-> Collector --> Emitter -|---            C           P           \n"
 "  ^   |                          |   | SCATTER                           \n"
 "  |   |      (ff_pipeline)       |   | (COMM1)                           \n"
 "  |    --------------------------    |              host1                \n"
 "  |          C            P          |       ----------------------      \n"
 "  | ALLGATHER                        |      |                      |     \n"
 "  |  (COMM2)                         |      |                      |     \n"
 "  |                                   ------|-> Worker --> Sender -|--   \n"
 "  |                                         |                      |  |  \n"
 "  |                                         |    (ff_pipeline)     |  |  \n"
 "  |                                          ----------------------   |  \n"
 "  |                                               C          P        |  \n"
 "   -------------------------------------------------------------------   \n"
 "                                                                         \n"
 "   COMM1, the server address is master:port1 (the address1 parameter)    \n"
 "   COMM2, the server address is master:port2 (the address2 parameter)    \n"
 "\n"); 
    return 0;
  }
    if (argc != 7) {
	std::cerr << "use: " << argv[0] << " matsize stream-length 1|0 nhosts masterhost:port1 masterhost:port2\n";
	std::cerr << "  1 for the master 0 for other hosts\n";
	std::cerr << "  nhosts: is the number of hosts for the master and the hostID for other hosts\n";
	std::cerr << "  masterhost: is the hostname or IP address of the master node.\n";

	return -1;
    }
    
    matSize = atoi(argv[1]);
    unsigned numTasks=atoi(argv[2]);
    char*    P = argv[3];                 // 1 for the master  0 for other hosts
    unsigned nhosts = atoi(argv[4]);      
    char*    address1  = argv[5];         // no check
    char*    address2  = argv[6];         // no check
        
    // creates the network using 0mq as transport layer
    zmqTransport transport(atoi(P)?-1 : nhosts);
    if (transport.initTransport()<0) abort();
    
    if (atoi(P)==1) { // master node
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
	printf("Master node Total Time= %.2f\n", C.ffTime());
    } else { // worker nodes
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
