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

 /**
 * \file perf_test_alloc2.cpp
 * \ingroup applications low-level-tests
 *
 * \brief \ref ff::ff_allocator and \ref ff::FFAllocator usage example
 *
 * This test gives the possibility to test different memory allocator
 * (libc, TBB, FastFlow, Hoard (compiling with USE_STANDARD and preloading the
 *  Hoard library) )
 *
 * @include perf_test_alloc2.cpp
 */

#include <sys/types.h>
//#include <unistd.h>
#include <stdlib.h>

#include <vector>
#include <iostream>
#include <ff/ff.hpp>
#include <ff/allocator.hpp>
#include <ff/cycle.h>
#if defined(USE_PROC_AFFINITY)
#include <ff/mapping_utils.hpp>
#endif


using namespace ff;

typedef unsigned long ff_task_t;
#if defined(USE_PROC_AFFINITY)
//WARNING: the following mapping targets dual-eight core Intel Sandy-Bridge 
const int worker_mapping[]   = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
const int collector_mapping  = 0;
const int PHYCORES           = 32;
#endif


#if defined(USE_TBB)
#include <tbb/scalable_allocator.h>
static tbb::scalable_allocator<char> * tbballocator=0;
#define ALLOCATOR_INIT() {tbballocator=new tbb::scalable_allocator<char>();}
#define MALLOC(size)   (tbballocator->allocate(size))
#define FREE(ptr,size) (tbballocator->deallocate((char *)ptr,size))

#elif defined(USE_CACHED_TBB)
#include <tbb/cache_aligned_allocator.h>
static tbb::cache_aligned_allocator<char> * tbballocator=0;
#include <tbb/cache_aligned_allocator.h>
#define ALLOCATOR_INIT() {tbballocator=new tbb::cache_aligned_allocator<char>();}
#define MALLOC(size)   (tbballocator->allocate(size))
#define FREE(ptr,size) (tbballocator->deallocate((char*)ptr,size))

#elif defined(USE_STANDARD)
/* standard libc malloc/free */
#define ALLOCATOR_INIT()
#define MALLOC(size)    malloc(size)
#define FREE(ptr,size)  free(ptr)

#else  /* FastFlow's allocator */
#define FF_ALLOCATOR 1
#include <ff/allocator.hpp>
#define ALLOCATOR_INIT() 
#define MALLOC(size)   (FFAllocator::instance()->malloc(size))

#if defined(DONT_USE_FFA)
static ff_allocator* MYALLOC[MAX_NUM_THREADS]={0};
#define FREE(ptr,id)  (MYALLOC[id]->free(ptr))
#else
#define FREE(ptr,unused) (FFAllocator::instance()->free(ptr))
#endif //DONT_USE_FFA

#endif 


// uncomment the following line to test FFAllocator
//#define TEST_FFA_MALLOC 1

#if defined(TEST_FFA_MALLOC)
class Worker: public ff_node {
private:
      void do_work(ff_task_t * task, int size, long long nticks) {
        for(int i=0;i<size;++i)
            task[i]=i;
        
        ticks_wait(nticks);
    }
public:
    Worker(int itemsize, int ntasks, long long nticks):
        itemsize(itemsize),ntasks(ntasks),nticks(nticks) {}

    void * svc(void *) {
        ff_task_t * task;
        for(int i=0;i<ntasks;++i) {
            // allocates memory
            task = (ff_task_t*)MALLOC(itemsize*sizeof(ff_task_t));
            bzero(task,itemsize*sizeof(ff_task_t));

            do_work(&task[0],itemsize,nticks);

            ff_send_out(task);
        }
        return NULL; 
    }
private:
    int            itemsize;
    int            ntasks;
    long long      nticks;
};


#else  // in this case we test only FFA's frees

// you have 2 possibilities:
#if 1 //this one ...
class Worker: public ff_node  {
private:
      void do_work(ff_task_t * task, int size, long long nticks) {
        for(int i=0;i<size;++i)
            task[i]=i;
        
        ticks_wait(nticks);
    }
public:
    Worker(int itemsize, int ntasks, long long nticks):
        myalloc(NULL),itemsize(itemsize),ntasks(ntasks),nticks(nticks) {

        myalloc=new ff_allocator();		      
        int slab = myalloc->getslabs(itemsize*sizeof(ff_task_t));
        int nslabs[N_SLABBUFFER];               
        if (slab<0) {                           
            if (myalloc->init()<0) abort();     
        } else {                                
            for(int i=0;i<N_SLABBUFFER;++i) {     
                if (i==slab) nslabs[i]=8192;      
                else nslabs[i]=0;                 
            }                
            if (myalloc->init(nslabs)<0) abort(); 
        }                                       
    }

    ~Worker() {
        if (myalloc) delete myalloc;
    }
    
    // called just one time at the very beginning
    int svc_init() {
#if defined(USE_PROC_AFFINITY)
        if (ff_mapThreadToCpu(worker_mapping[get_my_id() % PHYCORES])!=0)
            printf("Cannot map Worker %d CPU %d\n",get_my_id(),
                   worker_mapping[get_my_id() % PHYCORES]);
        //else printf("Thread %d mapped to CPU %d\n",get_my_id(), worker_mapping[get_my_id() % PHYCORES]);                                         
#endif

        // create a per-thread allocator
#if defined(FF_ALLOCATOR)
        myalloc->registerAllocator();
#if defined(DONT_USE_FFA)
        MYALLOC[get_my_id()]=myalloc;
#endif
#endif
        return 0;
    }

    void * svc(void *) {
        ff_task_t * task;
        for(int i=0;i<ntasks;++i) {
            // allocates memory
#if defined(FF_ALLOCATOR)
            task= (ff_task_t*)myalloc->malloc(itemsize*sizeof(ff_task_t));
#else
            task = (ff_task_t*)MALLOC(itemsize*sizeof(ff_task_t));
#endif
            memset(task,0,itemsize*sizeof(ff_task_t));

            do_work(&task[0],itemsize,nticks);            

            ff_send_out(task);
        }
        return NULL; 
    }

private:
    ff_allocator   * myalloc;
    int              itemsize;
    int              ntasks;
    long long        nticks;
};

#else  // and this one.

class Worker: public ff_node {
public:
    Worker(int itemsize, int ntasks,long long nticks):
        myalloc(NULL),itemsize(itemsize),ntasks(ntasks),nticks(nticks) {
    }

    ~Worker() {
        if (myalloc)
            FFAllocator::instance()->deleteAllocator(myalloc);
    }
    
    // called just one time at the very beginning
    int svc_init() {
#if defined(USE_PROC_AFFINITY)
        if (ff_mapThreadToCpu(worker_mapping[get_my_id() % PHYCORES])!=0)
            printf("Cannot map Worker %d CPU %d\n",get_my_id(),
                   worker_mapping[get_my_id() % PHYCORES]);
        //else printf("Thread %d mapped to CPU %d\n",get_my_id(), worker_mapping[get_my_id() % PHYCORES]);                                         
#endif
        // create a per-thread allocator
#if defined(FF_ALLOCATOR)
        myalloc= FFAllocator::instance()->newAllocator();
        if (!myalloc) {
            error("Worker, newAllocator fails\n");
            return -1;
        }
#endif
        return 0;
    }

    void * svc(void *) {
        ff_task_t * task;
        for(int i=0;i<ntasks;++i) {
            // allocates memory
#if defined(FF_ALLOCATOR)
            task= (ff_task_t*)myalloc->malloc(itemsize*sizeof(ff_task_t));
#else
            task = (ff_task_t*)MALLOC(itemsize*sizeof(ff_task_t));
#endif
            bzero(task,itemsize*sizeof(ff_task_t));

            do_work(&task[0],itemsize,nticks);

            ff_send_out(task);
        }
        return NULL; 
    }

private:
    ffa_wrapper     * myalloc;
    int              itemsize;
    int              ntasks;
    long long        nticks;
};

#endif
#endif 



// it just starts all workers than it exits
class Emitter: public ff_monode {
public:
    void * svc(void * ) {
        std::cerr << "Emitter received task\n";
        broadcast_task(GO_ON);
        return EOS; 
    }
};

class Collector: public ff_minode {
public:
    Collector(int itemsize):itemsize(itemsize) {}
    
    int svc_init() {
#if defined(USE_PROC_AFFINITY)
        if (ff_mapThreadToCpu(collector_mapping)!=0)
            printf("Cannot map Collector to CPU %d\n",collector_mapping);
        //else  printf("Collector mapped to CPU %d\n", collector_mapping);                                                                                                                                         
#endif
#if !defined(USE_TBB)
        itemsize = 0;          // just to avoid warnings
#endif
#if !defined(DONT_USE_FFA)
        get_channel_id();  // just to avoid warnings
#endif
        return 0;
    }

    // Even in this case we have 2 options:
    // 1. to use the FFAllocator (the simplest option)
    // 2. to use a local data structure which mantains the association between
    //    the allocator and the worker id (you must define DONT_USE_FFA for this option)
    void * svc(void * task) { 
#if defined(DONT_USE_FFA)
        FREE(task, get_channel_id());
#else
        // the size is required for TBB's allocator
        FREE(task,itemsize*sizeof(ff_task_t));
#endif
        return GO_ON; 
    }
private:
    int itemsize; // needed for TBB's allocators
};

int main(int argc, char * argv[]) {    
    unsigned int ntasks         = 1000000;
    unsigned int itemsize       = 16;
    int nworkers                = 3;
    long long    nticks         = 1000;
    if (argc>1) {
        if (argc<5) {
            std::cerr 
                << "use: "  << argv[0] 
                << " ntasks num-integer-x-item #n nticks\n";
            return -1;
        }
        
        ntasks         = atoi(argv[1]);
        itemsize       = atoi(argv[2]);
        nworkers       = atoi(argv[3]);    
        nticks         = strtoll(argv[4],NULL,10);
    }
	std::cerr << "ticks " << nticks << "\n";
	// arguments check
    if (nworkers<0 || !ntasks) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    ALLOCATOR_INIT();

    if (nworkers==0) {
        ffTime(START_TIME);
        for(unsigned int i=0;i<ntasks;++i) {
            // allocates memory
            void * task= MALLOC(itemsize*sizeof(ff_task_t));
            memset(task,0,itemsize*sizeof(ff_task_t));
            FREE(task,itemsize*sizeof(ff_task_t));
        }
        ffTime(STOP_TIME);       
        std::cerr << "DONE, time= " << ffTime(GET_TIME) << " (ms)\n";
        return 0;
    } 

    // create the farm object
    ff_farm farm;
    // create and add emitter object to the farm
    Emitter E;
    farm.add_emitter(&E);
    
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) 
        w.push_back(new Worker(itemsize,ntasks/nworkers,nticks));
    farm.add_workers(w);
        
    Collector C(itemsize);
    farm.add_collector(&C);
    
    // let's start
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
        
    std::cerr << "DONE, time= " << farm.ffTime() << " (ms)\n";
    return 0;
}
