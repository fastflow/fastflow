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
 * Multiple malloc threads, one free thread.
 *
 */

#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include <vector>
#include <iostream>
#include <farm.hpp>
#include <allocator.hpp>
#include <cycle.h>

using namespace ff;

typedef int task_t;


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
#include <allocator.hpp>
#define ALLOCATOR_INIT() 
#define MALLOC(size)   (FFAllocator::instance()->malloc(size))
#define FREE(ptr,size) (FFAllocator::instance()->free(ptr))

#endif

// comment the following line to use a different
// implementation
#define TEST_FFA_MALLOC

#if defined(TEST_FFA_MALLOC)
class Worker: public ff_node {
public:
    Worker(int itemsize, int ntasks):
        itemsize(itemsize),ntasks(ntasks) {}

    void * svc(void * task) {
        for(int i=0;i<ntasks;++i) {
            // allocates memory
            task= MALLOC(itemsize*sizeof(task_t));
            bzero(task,itemsize*sizeof(task_t));

            ff_send_out(task);
        }
        return NULL; 
    }
private:
    int            itemsize;
    int            ntasks;
};

#else  // in this case we test only FFA's frees

class Worker: public ff_node {
public:
    Worker(int itemsize, int ntasks):
        myalloc(NULL),itemsize(itemsize),ntasks(ntasks) {}

    ~Worker() {
        if (myalloc)
            FFAllocator::instance()->deleteAllocator(myalloc);
    }
    
    // called just one time at the very beginning
    int svc_init() {
        // get a per-thread allocator
        myalloc= FFAllocator::instance()->newAllocator();
        if (!myalloc) {
            error("Worker, newAllocator fails\n");
            return -1;
        }
        return 0;
    }

    void * svc(void * task) {
        for(int i=0;i<ntasks;++i) {
            // allocates memory
            task= myalloc->malloc(itemsize*sizeof(task_t));
            bzero(task,itemsize*sizeof(task_t));

            ff_send_out(task);
        }
        return NULL; 
    }

private:
    ffa_wrapper  * myalloc;
    int            itemsize;
    int            ntasks;
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

class Collector: public ff_node {
public:
    Collector(int itemsize):itemsize(itemsize) {}
    void * svc(void * task) { 
        FREE(task,itemsize*sizeof(task_t));
        return GO_ON; 
    }
private:
    int itemsize; // needed for TBB's allocators
};

int main(int argc, char * argv[]) {    
    if (argc<4) {
        std::cerr 
            << "use: "  << argv[0] 
            << " ntasks num-integer-x-item #n\n";
        return -1;
    }
    
    unsigned int ntasks         = atoi(argv[1]);
    unsigned int itemsize       = atoi(argv[2]);
    unsigned int nworkers       = atoi(argv[3]);    

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
            void * task= MALLOC(itemsize*sizeof(task_t));
            bzero(task,itemsize*sizeof(task_t));
            FREE(task,itemsize*sizeof(task_t));
        }
        ffTime(STOP_TIME);       
        std::cerr << "DONE, time= " << ffTime(GET_TIME) << " (ms)\n";
        return 0;
    } 
    // create the farm object
    ff_farm<> farm;
    std::vector<ff_node *> w;
    for(unsigned int i=0;i<nworkers;++i) 
        w.push_back(new Worker(itemsize,ntasks/nworkers));
    farm.add_workers(w);
    
    
    // create and add emitter object to the farm
    Emitter E(nworkers);
    farm.add_emitter(&E);
    
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
