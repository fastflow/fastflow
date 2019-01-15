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
 * \file perf_test_alloc1.cpp
 * \ingroup applications low-level-tests
 *
 * \brief \ref ff::ff_allocator usage example
 *
 * This test gives the possibility to test different memory allocator
 * (libc, TBB, FastFlow, Hoard (compiling with USE_STANDARD and preloading the 
 *  Hoard library) )
 *
 * @include perf_test_alloc1.cpp
 */

#include <sys/types.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <sstream>
#include <string>
#include <ff/ff.hpp>
#include <ff/cycle.h>
#if defined(USE_PROC_AFFINITY)
#include <ff/mapping_utils.hpp>
#endif

using namespace ff;

typedef unsigned long ff_task_t;
#if defined(USE_PROC_AFFINITY)
//WARNING: the following mapping targets dual-eight core Intel Sandy-Bridge 
const int worker_mapping[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
const int emitter_mapping  = 0;
const int PHYCORES         = 16;
#endif

#if defined(USE_TBB)
#include <tbb/scalable_allocator.h>
static tbb::scalable_allocator<char> * tbballocator=0;
#define ALLOCATOR_INIT(size) {tbballocator=new tbb::scalable_allocator<char>();}
#define MALLOC(size)   (tbballocator->allocate(size))
#define FREE(ptr,size) (tbballocator->deallocate((char *)ptr,size))

#elif defined(USE_CACHED_TBB)
#include <tbb/cache_aligned_allocator.h>
static tbb::cache_aligned_allocator<char> * tbballocator=0;
#include <tbb/cache_aligned_allocator.h>
#define ALLOCATOR_INIT(size) {tbballocator=new tbb::cache_aligned_allocator<char>();}
#define MALLOC(size)   (tbballocator->allocate(size))
#define FREE(ptr,size) (tbballocator->deallocate((char*)ptr,size))

#elif defined(USE_STANDARD)
/* standard libc malloc/free */
#define ALLOCATOR_INIT(size)
#define MALLOC(size)    malloc(size)
#define FREE(ptr,size)  free(ptr)

#else  /* FastFlow's allocator */
#define FF_ALLOCATOR 1
#include <ff/allocator.hpp>
static ff_allocator * ffalloc = 0;

/* default init 
#define ALLOCATOR_INIT(size)  {    \
 ffalloc=new ff_allocator();	   \
 if (ffalloc->init()<0) abort();   \
}
*/
#define ALLOCATOR_INIT(size) {					\
        ffalloc=new ff_allocator();		        \
        int slab = ffalloc->getslabs(size);     \
        int nslabs[N_SLABBUFFER];               \
        if (slab<0) {                           \
            if (ffalloc->init()<0) abort();     \
        } else {                                \
          for(int i=0;i<N_SLABBUFFER;++i) {     \
              if (i==slab) nslabs[i]=8192;      \
              else nslabs[i]=0;                 \
          }                                     \
          if (ffalloc->init(nslabs)<0) abort(); \
        }                                       \
     }

#define MALLOC(size)   (ffalloc->malloc(size))
#define FREE(ptr,size) (ffalloc->free(ptr))

#endif




// generic worker
class Worker: public ff_node {
protected:
    void do_work(ff_task_t * task, int size, long long nticks) {
        for(int i=0;i<size;++i) {
            task[i]+=1;

            // task[i]+=sin(1.0/(task[i]));
            // for(int j=0;j<3;++j) {
            //     double a =(sin(1.0/cos(task[i]) + 1.12321321*sin(task[i]))* cos(task[i])/sin(task[i]));
            //     task[i] += j+ ((double)a*a/3.14 - 12321321.1231);
            // }
        }
        
        ticks_wait(nticks);
    }

public:
    Worker(int itemsize, long long nticks):itemsize(itemsize),nticks(nticks) {}

    // called just one time at the very beginning
    int svc_init() {
#if defined(FF_ALLOCATOR)        
        if (ffalloc->register4free()<0) {
            error("Worker, register4free fails\n");
            return -1;
        }
#endif
#if defined(USE_PROC_AFFINITY)
        if (ff_mapThreadToCpu(worker_mapping[get_my_id() % PHYCORES])!=0)
            printf("Cannot map Worker %d CPU %d\n",get_my_id(),
                   worker_mapping[get_my_id() % PHYCORES]);
        //else printf("Thread %d mapped to CPU %d\n",get_my_id(), worker_mapping[get_my_id() % PHYCORES]);                                                                                                     
#endif
        return 0;
    }

    void * svc(void * t) {
        ff_task_t * task = (ff_task_t *)t;
        do_work(&task[1],itemsize-1,nticks);
        FREE(task,itemsize);
        // we don't have the collector so we have any task to send out
        return GO_ON; 
    }

private:
    int itemsize;
    long long nticks;
};


// the load-balancer filter
class Emitter: public ff_node {
    int val;
protected:

    inline void filltask(ff_task_t * task, size_t size) {
        ++val;
        for(unsigned int i=0;i<size;++i)
            task[i]=val;
    }

public:
    Emitter(int max_task, int itemsize):ntask(max_task),itemsize(itemsize) {
        ALLOCATOR_INIT(itemsize*sizeof(ff_task_t));
        val=0;
    };

    // called just one time at the very beginning
    int svc_init() {
#if defined(FF_ALLOCATOR)
        if (ffalloc->registerAllocator()<0) {
            error("Emitter, registerAllocator fails\n");
            return -1;
        }
#endif
#if defined(USE_PROC_AFFINITY)
        if (ff_mapThreadToCpu(emitter_mapping)!=0)
            printf("Cannot map Emitter to CPU %d\n",emitter_mapping);
        //else  printf("Emitter mapped to CPU %d\n", emitter_mapping);                                                                                                                                         
#endif
        return 0;
    }

    void * svc(void *) {
        ff_task_t * task = (ff_task_t*)MALLOC(itemsize*sizeof(ff_task_t));
        if (!task) abort();
        filltask(&task[1], itemsize-1);
        task[0] = 0;
        --ntask;
        if (ntask<0) {
            FREE(task,itemsize);
            return NULL;
        }
        return task;
    }
private:
    int ntask;
    int itemsize;
};


int main(int argc, char * argv[]) {    
    unsigned int buffer_entries = 128;
    unsigned int streamlen      = 1000000;
    unsigned int itemsize       = 32;
    int nworkers                = 3;
    long long nticks            = 1000;

    if (argc>1) {
        if (argc<6) {
            std::cerr 
                << "use: "  << argv[0] 
                << " num-buffer-entries streamlen num-integer-x-item #n nticks\n";
            return -1;
        }
        
        buffer_entries = atoi(argv[1]);
        streamlen      = atoi(argv[2]);
        itemsize       = atoi(argv[3]);
        nworkers       = atoi(argv[4]);    
        nticks         = atol(argv[5]);
    }
    // arguments check
    if (nworkers<0 || !streamlen || nticks<0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    if (nworkers==0) {
        Emitter E(streamlen, itemsize);
        Worker  W(itemsize, nticks);
        
        ffTime(START_TIME);
        E.svc_init();
        W.svc_init();
        do {
            void * t = E.svc(NULL);
            if (!t) break;
            W.svc(t);
        } while(1);
        ffTime(STOP_TIME);       

        std::cerr << "DONE, time= " << ffTime(GET_TIME) << " (ms)\n";
        return 0;
    } 
    // create the farm object
    ff_farm farm(false, buffer_entries*nworkers);
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) 
        w.push_back(new Worker(itemsize,nticks));
    farm.add_workers(w);
    
    // create and add to the farm the emitter object
    Emitter E(streamlen, itemsize);
    farm.add_emitter(&E);
    
    // let's start
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }

    std::cerr << "DONE, time= " << farm.ffTime() << " (ms)\n";
    farm.ffStats(std::cout);
#if defined(FF_ALLOCATOR) && defined(ALLOCATOR_STATS)
    ffalloc->deregisterAllocator();
    ffalloc->printstats(std::cout);
#endif

    return 0;
}
