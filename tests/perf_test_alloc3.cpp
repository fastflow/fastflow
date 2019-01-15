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
 * Tests per-thread allocator. 
 * Each thread allocates and frees memory.
 *
 * use:
 *   perf_test_alloc3 10000000 8 10000 #P
 *     - where #P is the number of threads
 *     - 8 is equivalent to 8*sizeof(long) bytes
 */

#include <sys/types.h>
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
const int worker_mapping[]   = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
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
#if defined(TEST_FFA_MALLOC)
#define MALLOC(size)   (FFAllocator::instance()->malloc(size))
#define FREE(ptr,size) (FFAllocator::instance()->free(ptr))
#else
#define MALLOC(size)   (myalloc->malloc(size))
#define FREE(ptr,size) (myalloc->free(ptr))
#endif // TEST_FFA_MALLOC
#endif

// un-comment the following line to use a different
// implementation
//#define TEST_FFA_MALLOC

#if defined(TEST_FFA_MALLOC)
class Worker: public ff_node {
public:
    Worker(int itemsize, int ntasks, int batchsize):
        itemsize(itemsize),ntasks(ntasks),batchsize(batchsize) {}

    void * svc(void *) {
        ff_task_t* task[batchsize];

        int numbatch = ntasks/batchsize;
        for(int j=0;j<numbatch;++j) {              
            for(int i=0;i<batchsize;++i) {
                task[i]= (ff_task_t*)MALLOC(itemsize*sizeof(ff_task_t));
                memset(task[i],0,itemsize*sizeof(ff_task_t));
            }
            for(int i=0;i<batchsize;++i) {
                FREE(task[i],itemsize*sizeof(ff_task_t));
            }
            ntasks-=batchsize;
        }
        for(int i=0;i<ntasks;++i) {
            task[i]= (ff_task_t*)MALLOC(itemsize*sizeof(ff_task_t));
            memset(task[i],0,itemsize*sizeof(ff_task_t));
        }
        for(int i=0;i<ntasks;++i) {
            FREE(task[i],itemsize*sizeof(ff_task_t));
        }
        return NULL; 
    }
private:
    int            itemsize;
    int            ntasks;
    int            batchsize;
};

#else  

// static inline void* aligned_malloc(size_t sz)
// {
//     void*               mem;
//     if (posix_memalign(&mem, 128, sz))
//         return 0;
//     return mem;
// }


class Worker: public ff_node {
public:
    Worker(int itemsize, int ntasks, int batchsize):
        myalloc(NULL),itemsize(itemsize),ntasks(ntasks),batchsize(batchsize) {}
    
    // called just one time at the very beginning
    int svc_init() {
#if defined(USE_PROC_AFFINITY)
        if (ff_mapThreadToCpu(worker_mapping[get_my_id() % PHYCORES])!=0)
            printf("Cannot map Worker %d CPU %d\n",get_my_id(),
                   worker_mapping[get_my_id() % PHYCORES]);
        //else printf("Thread %d mapped to CPU %d\n",get_my_id(), worker_mapping[get_my_id() % PHYCORES]);                                         
#endif
        
#if defined(FF_ALLOCATOR)
        myalloc=new(malloc(sizeof(ff_allocator))) ff_allocator();		      
        if (myalloc->registerAllocator()<0) {
            error("Worker, registerAllocator fails\n");
            return -1;
        }
        int slab = myalloc->getslabs(itemsize*sizeof(ff_task_t));
        int nslabs[N_SLABBUFFER];               
        if (slab<0) {                           
            if (myalloc->init()<0) abort();     
        } else {                                
            for(int i=0;i<N_SLABBUFFER;++i) {     
                if (i==slab) nslabs[i]=2*batchsize;
                else nslabs[i]=0;                 
            }                
            if (myalloc->init(nslabs)<0) abort(); 
        }
#endif
        return 0;
    }

    void * svc(void *) {
        //ff_task_t* task[batchsize];
		ff_task_t **task;
		task = (ff_task_t **) MALLOC(batchsize*sizeof(ff_task_t *));

        int numbatch = ntasks/batchsize;
        for(int j=0;j<numbatch;++j) {              
            for(int i=0;i<batchsize;++i) {
                task[i]= (ff_task_t*)MALLOC(itemsize*sizeof(ff_task_t));
                memset(task[i],0,itemsize*sizeof(ff_task_t));
            }
            for(int i=0;i<batchsize;++i) {
                FREE(task[i],itemsize*sizeof(ff_task_t));
            }
            ntasks-=batchsize;
        }
        for(int i=0;i<ntasks;++i) {
            task[i]= (ff_task_t*)MALLOC(itemsize*sizeof(ff_task_t));
            memset(task[i],0,itemsize*sizeof(ff_task_t));
        }
        for(int i=0;i<ntasks;++i) {
            FREE(task[i],itemsize*sizeof(ff_task_t));
        }
		FREE(task,batchsize*sizeof(ff_task_t *));
        return NULL; 
	}

    void svc_end() {
        //if (myalloc) FFA->deleteAllocator(myalloc);
        if (myalloc) delete myalloc;
#if defined(FF_ALLOCATOR) && defined(ALLOCATOR_STATS)
        myalloc->deregisterAllocator();
        myalloc->printstats(std::cout);
#endif
    }

private:
    ff_allocator * myalloc;
    int            itemsize;
    int            ntasks;
    int            batchsize;
};
#endif 


// it just starts all workers.....
class Emitter: public ff_node {
public:
    Emitter(int nworkers):nworkers(nworkers) {}
    void * svc(void * task) {
        task = GO_ON;
        for(int i=0;i<nworkers;++i)
            ff_send_out(task);

        return NULL; 
    }
private:
    int nworkers;
};


int main(int argc, char * argv[]) {    
    unsigned int ntasks         = 10000000;
    unsigned int itemsize       = 32;
    unsigned int batch          = 128;
    int nworkers                = 3;
    if (argc>1) {
        if (argc<5) {
            std::cerr 
                << "use: "  << argv[0] 
                << " ntasks num-integer-x-item batchsize #n\n";
            return -1;
        }
    
        ntasks         = atoi(argv[1]);
        itemsize       = atoi(argv[2]);
        batch          = atoi(argv[3]);    
        nworkers       = atoi(argv[4]);    
    }
    // arguments check
    if (nworkers<0 || !ntasks) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    ALLOCATOR_INIT();

    if (nworkers==0) {
        ffTime(START_TIME);
        Worker* w = new Worker(itemsize,ntasks,batch);
        w->svc_init();
        w->svc(NULL);
        w->svc_end();
        ffTime(STOP_TIME);
        std::cerr << "DONE, seq time= " << ffTime(GET_TIME) << " (ms)\n";        
        return 0;
    }


    // create the farm object
    ff_farm farm;
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) 
        w.push_back(new Worker(itemsize,ntasks/nworkers, batch));
    farm.add_workers(w);
    
    
    // create and add emitter object to the farm
    Emitter E(nworkers);
    farm.add_emitter(&E);
        
    // let's start
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    std::cerr << "DONE, time= " << farm.ffTime() << " (ms)\n";
    return 0;
}
