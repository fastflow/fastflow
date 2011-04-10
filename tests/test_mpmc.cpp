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
 *  December 2010
 *  April    2011  (major rework for the MSqueue)
 *
 */
#include <iostream>
#include <ff/node.hpp>   // for Barrier
#include <ff/cycle.h>
#include <ff/atomic/atomic.h>
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
multiMSqueueCDS * msq;
#else
#if defined(SCALABLE_QUEUE)
multiMSqueue    * msq;
#else // !SCALABLE_QUEUE
MSqueue         * msq;
#endif // SCALABLE_QUEUE
#endif // HAVE_CDSLIB
#endif // USE_LFDS

const int MAX_NUM_THREADS=128;  // just an upper bound, it can be increased

#if defined(SCALABLE_QUEUE) || defined(HAVE_CDSLIB)
int nqueues=4;
#endif
int ntasks=0;                   // total number of tasks
bool end = false;               // termination flag
atomic_long_t counter;           
std::vector<long> results;

// for statistics
long taskC[MAX_NUM_THREADS]={0};
long taskP[MAX_NUM_THREADS]={0};

static inline bool PUSH(int myid) {
    long * p = (long *)(atomic_long_inc_return(&counter));

    if ((long)p > ntasks) return false;

#if defined(USE_LFDS)
    do ; while( !queue_enqueue(msq, p ) );
#else
    do ; while(!(msq->push(p)));
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

    Barrier::instance()->barrier();
    do; while(PUSH(myid));

#if defined(HAVE_CDSLIB)
    cds::threading::pthread::Manager::detachThread();
#endif
    pthread_exit(NULL);
}
    
// consumer function
void * C(void * arg) {
    int myid= *(int*)arg;

    union {long a; void *b;} task;
    task.b=NULL;

#if defined(HAVE_CDSLIB)
    cds::threading::pthread::Manager::attachThread();
#endif

    Barrier::instance()->barrier();

    while(!end) {
        if (QUEUE_POP(task.b))  {
            if (task.b == (void*)FF_EOS) {
                end=true;
            }
            else {
                if (task.a > ntasks) {
                    std::cerr << "received " << task.a << " ABORT\n";
                    abort();
                }
                results[task.a-1] = task.a;
                ++taskC[myid];
            }
        } 
    }
    ffTime(STOP_TIME);
    
#if defined(HAVE_CDSLIB)
    cds::threading::pthread::Manager::detachThread();
#endif
    pthread_exit(NULL);
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
    msq = new multiMSqueueCDS(nqueues);
 #else
  #if defined(SCALABLE_QUEUE)
    msq = new multiMSqueue(nqueues);
  #else
    msq = new MSqueue;
    assert(msq->init());
  #endif
 #endif
#endif

    atomic_long_set(&counter,0);

    pthread_t P_handle[numP], C_handle[numC];

    // define the number of threads that are going to partecipate....
    Barrier::instance()->barrier(numP+numC);

    ffTime(START_TIME);

    int idC[numC];
    for(int i=0;i<numC;++i) {
        idC[i]=i;
        if (pthread_create(&C_handle[i], NULL,C,&idC[i]) != 0) {
            abort();
        }
    }
    int idP[numP];
    for(int i=0;i<numP;++i)  {
        idP[i]=i;
        if (pthread_create(&P_handle[i], NULL,P,&idP[i]) != 0) {
            abort();
        }
    }
    
    // wait all producers
    for(int i=0;i<numP;++i) {
        pthread_join(P_handle[i],NULL);
    }

    // send EOS to stop the consumers
#if defined(SCALABLE_QUEUE) || defined(HAVE_CDSLIB)
    msq->stop_producing();
#else
    for(int i=0;i<numC;++i) {
#if defined(USE_LFDS)
        do ; while(! queue_enqueue(msq, (void*)FF_EOS));
#else
        do ; while(! msq->push((long*)FF_EOS)); 
#endif
    }
#endif

    // wait all consumers
    for(int i=0;i<numC;++i) {
        pthread_join(C_handle[i],NULL);
    }

    std::cout << "Checking result...\n";
    // check result
    bool wrong = false;
    for(int i=0;i<ntasks;++i)
        if (results[i] != i+1) {
            std::cerr << "WRONG result " << results[i] << " should be " << i+i << "\n";
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
