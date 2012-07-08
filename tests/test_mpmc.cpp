/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */
/*
 * Simple test for the Multi-Producer/Multi-Consumer MSqueue.
 *
 * Author: Massimo Torquati
 *  December 2010  first version
 *  April    2011  (major rework for the MSqueue)
 *  April    2012  added the bounded/unbounded MPMC queues 
 *
 */
#include <iostream>
#include <ff/node.hpp>   // for Barrier
#include <ff/cycle.h>
#include <ff/atomic/atomic.h>
#include <ff/platforms/platform.h>
//#include <ff/mapping_utils.hpp>

using namespace ff;

#if defined(USE_LFDS)
// you need to install liblfds if you want to 
// test the MSqueue implementation contained
// in that library - www.liblfds.org
extern "C" {
#include <liblfds.h>
};
struct queue_state *msq;
#else  // !USE_LDFS

#include <ff/MPMCqueues.hpp>

#if defined(HAVE_CDSLIB)
 #if defined(SCALABLE_QUEUE)
  multiMSqueueCDS * msq;
 #else
#include <cds/container/msqueue.h>
#include <cds/gc/hp.h>
cds::container::MSQueue< cds::gc::HP, void*> * msq;  // which allocator ?
 #endif
#else
 #if defined(SCALABLE_QUEUE)
  multiMSqueue    * msq;
 #else // !SCALABLE_QUEUE
  #if defined(BOUNDED_MPMC)
   MPMC_Ptr_Queue  * msq;
  #else
   #if defined(UNBOUNDED_MPMC)
    uMPMC_Ptr_Queue * msq;
   #else
    MSqueue         * msq;
   #endif // UNBOUNDED_MPMC
  #endif // BOUNDED_MPMC
 #endif // SCALABLE_QUEUE
#endif // HAVE_CDSLIB
#endif // USE_LFDS

const int MAX_NUM_THREADS=128;  // just an upper bound, it can be increased

#if defined(SCALABLE_QUEUE) || defined(HAVE_CDSLIB) || defined(UNBOUNDED_MPMC)
int nqueues=4;
#endif
int ntasks=0;                   // total number of tasks
atomic_long_t counter;           
std::vector<long> results;

// for statistics
long taskC[MAX_NUM_THREADS]={0};
long taskP[MAX_NUM_THREADS]={0};



/* 
 * The following function has been taken from CDS library 
 * (http://sourceforge.net/projects/libcds/).
 * 
 * FIX: It works only for gcc compiler and x86 and x86_64 architectures.
 *
 */
#if (defined __GNUC__ && (defined __i686__ || defined __i386__ || defined__x86_64__))
static inline void backoff_pause( unsigned int nLoop = 0x000003FF ) {
    asm volatile (
                  "andl %[nLoop], %%ecx;      \n\t"
                  "cmovzl %[nLoop], %%ecx;    \n\t"
                  "rep; "
                  "nop;   \n\t"
                  : /*no output*/
                  : [nLoop] "r" (nLoop)
                  : "ecx", "cc"
                  )    ;
}
#else
static inline void backoff_pause( unsigned int nLoop = 0x000003FF ) {
    return;
}
#endif


static inline bool PUSH(int myid) {
    long * p = (long *)(atomic_long_inc_return(&counter));

    if ((long)p > ntasks) return false;

#if defined(USE_LFDS)
    do ; while( !queue_enqueue(msq, p ) );
#else
    void * p2= p;
    do ; while(!(msq->push(p2)));
#endif

    ++taskP[myid];

    return true;
}   

#if defined(USE_LFDS)
 #define QUEUE_POP(x) queue_dequeue(msq, &x)
#else
 #define QUEUE_POP(x) msq->pop(x)
#endif

// producer function
void * P(void * arg) {
    int myid = *(int*)arg;

#if defined(HAVE_CDSLIB)
    cds::threading::pthread::Manager::attachThread();
#endif

    Barrier::instance()->doBarrier();
    do; while(PUSH(myid));

#if defined(HAVE_CDSLIB)
    cds::threading::pthread::Manager::detachThread();
#endif
    pthread_exit(NULL);
	return NULL;
}
    
// consumer function
void * C(void * arg) {
    int myid= *(int*)arg;

    union {long a; void *b;} task;
    task.b=NULL;

#if defined(HAVE_CDSLIB)
    cds::threading::pthread::Manager::attachThread();
#endif

    Barrier::instance()->doBarrier();
    while(1) {
        if (!QUEUE_POP(&task.b))  {
            backoff_pause();
            continue;
        }
        if (task.b == (void*)FF_EOS) break;

        if (task.a > ntasks) {
            std::cerr << "received " << task.a << " ABORT\n";
            abort();
        }
        results[task.a-1] = task.a;
        ++taskC[myid];
    }

    ffTime(STOP_TIME,true);
    
#if defined(HAVE_CDSLIB)
    cds::threading::pthread::Manager::detachThread();
#endif
    pthread_exit(NULL);
	return NULL;
}

#include <ff/mapping_utils.hpp>

int main(int argc, char * argv[]) {
#if defined(HAVE_CDSLIB)
    cds::gc::hzp::GarbageCollector::Construct();
    cds::threading::pthread::Manager::init();
    cds::threading::pthread::Manager::attachThread();
#endif

    if (argc<4) {
        std::cerr << "use: " << argv[0] << " ntasks #P #C [nqueues]\n";        
        return -1;
    }
    ntasks= atoi(argv[1]);
    int numP  = atoi(argv[2]);
    int numC  = atoi(argv[3]);
    if ((numP <=0) || (numC<=0)) {
        std::cerr << "Error: #P >0 #C >0 \n";
        return -1;
    }
#if defined(SCALABLE_QUEUE)
    if (argc>=5)
        nqueues = atoi(argv[4]);
#endif

    if (numP+numC > MAX_NUM_THREADS) {
        std::cerr << "too many threads, please increase MAX_NUM_THREADS\n";
        return -1;
    }

    results.resize(ntasks,-1);

#if defined(USE_LFDS)
    queue_new( &msq, 1000000 );
#else
 #if defined(HAVE_CDSLIB)
  #if defined(SCALABLE_QUEUE)
    msq = new multiMSqueueCDS(nqueues);
  #else
    msq = new cds::container::MSQueue< cds::gc::HP, void*>;
    if (!msq) abort();
  #endif
 #else
  #if defined(SCALABLE_QUEUE)
    msq = new multiMSqueue(nqueues);
  #else
   #if defined(BOUNDED_MPMC)
    msq = new MPMC_Ptr_Queue(ntasks/2); // we set the size to half ntasks
    if (!msq->init()) abort();
   #else
    #if defined(UNBOUNDED_MPMC)
     msq = new uMPMC_Ptr_Queue;
     if (!msq->init(nqueues, ntasks/(2*nqueues))) abort();
    #else
     msq = new MSqueue;
     if (!msq->init()) abort();
    #endif // UNBOUNDED_MPMC
   #endif // BOUNDED_MPMC
  #endif
 #endif
#endif

    atomic_long_set(&counter,0);

    pthread_t * P_handle, * C_handle;

	P_handle = (pthread_t *) malloc(sizeof(pthread_t)*numP);
	C_handle = (pthread_t *) malloc(sizeof(pthread_t)*numC);
	
    // define the number of threads that are going to partecipate....
    Barrier::instance()->doBarrier(numP+numC+1);

    int * idC;
	idC = (int *) malloc(sizeof(int)*numC);
    for(int i=0;i<numC;++i) {
        idC[i]=i;
        if (pthread_create(&C_handle[i], NULL,C,&idC[i]) != 0) {
            abort();
        }
    }
    int *idP;
	idP = (int *) malloc(sizeof(int)*numP);
    for(int i=0;i<numP;++i)  {
        idP[i]=i;
        if (pthread_create(&P_handle[i], NULL,P,&idP[i]) != 0) {
            abort();
        }
    }

    ffTime(START_TIME);
    Barrier::instance()->doBarrier();

    // wait all producers
    for(int i=0;i<numP;++i) {
        pthread_join(P_handle[i],NULL);
    }
    
    for(int i=0;i<numC;++i) {
#if defined(USE_LFDS)
        do ; while(! queue_enqueue(msq, (void*)FF_EOS));
#else
        do ; while(! msq->push((long*)FF_EOS)); 
#endif
    }

    // wait all consumers
    for(int i=0;i<numC;++i) {
        pthread_join(C_handle[i],NULL);
    }

    std::cout << "Checking result...\n";
    // check result
    bool wrong = false;
    for(int i=0;i<ntasks;++i)
        if (results[i] != i+1) {
            std::cerr << "WRONG result in position " << i << " is " << results[i] << " should be " << i+1 << "\n";
            wrong = true;
        }
    if (!wrong)  std::cout << "Ok. Done!\n";

    std::cerr << "Time " << ffTime(GET_TIME) << " (ms)\n";

#if 0
    // stats
    for(int i=0;i<numP;++i)
        std::cout << "P " << i << " got " << taskP[i] << " tasks\n";
    for(int i=0;i<numC;++i)
        std::cout << "C " << i << " got " << taskC[i] << " tasks\n";
#endif


    return 0;
}
