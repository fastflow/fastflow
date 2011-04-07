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
//#include <unistd.h>
#include <stdlib.h>

#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/allocator.hpp>
#include <ff/cycle.h>

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
#include <ff/allocator.hpp>
#define ALLOCATOR_INIT() 
#define MALLOC(size)   (FFAllocator::instance()->malloc(size))
#define FREE(ptr,size) (FFAllocator::instance()->free(ptr))

#endif

// uncomment the following line to test FFAllocator
//#define TEST_FFA_MALLOC 1

#if defined(TEST_FFA_MALLOC)
class Worker: public ff_node {
private:
      void do_work(task_t * task, int size, long long nticks) {
        for(register int i=0;i<size;++i)
            task[i]=i;
        
        ticks_wait(nticks);
    }
public:
    Worker(int itemsize, int ntasks, long long nticks):
        itemsize(itemsize),ntasks(ntasks),nticks(nticks) {}

    void * svc(void *) {
        task_t * task;
        for(int i=0;i<ntasks;++i) {
            // allocates memory
            task= (task_t*)MALLOC(itemsize*sizeof(task_t));
            bzero(task,itemsize*sizeof(task_t));

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
      void do_work(task_t * task, int size, long long nticks) {
        for(register int i=0;i<size;++i)
            task[i]=i;
        
        ticks_wait(nticks);
    }
public:
    Worker(int itemsize, int ntasks, long long nticks):
        myalloc(NULL),itemsize(itemsize),ntasks(ntasks),nticks(nticks) {

        myalloc = new ff_allocator();
        myalloc->init();
    }

    ~Worker() {
        if (myalloc) delete myalloc;
    }
    
    // called just one time at the very beginning
    int svc_init() {
        // create a per-thread allocator
#if defined(FF_ALLOCATOR)
        myalloc->registerAllocator();
#endif
        return 0;
    }

    void * svc(void *) {
        task_t * task;
        for(int i=0;i<ntasks;++i) {
            // allocates memory
#if defined(FF_ALLOCATOR)
            task= (task_t*)myalloc->malloc(itemsize*sizeof(task_t));
#else
            task = (task_t*)MALLOC(itemsize*sizeof(task_t));
#endif
            memset(task,0,itemsize*sizeof(task_t));

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
        task_t * task;
        for(int i=0;i<ntasks;++i) {
            // allocates memory
#if defined(FF_ALLOCATOR)
            task= (task_t*)myalloc->malloc(itemsize*sizeof(task_t));
#else
            task = (task_t*)MALLOC(itemsize*sizeof(task_t));
#endif
            bzero(task,itemsize*sizeof(task_t));

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


/* 
 * You have to extend the ff_loadbalancer....
 */
class my_loadbalancer: public ff_loadbalancer {
public:
    // this is necessary because ff_loadbalancer has non default parameters....
    my_loadbalancer(int max_num_workers):ff_loadbalancer(max_num_workers) {}

    void broadcast(void * task) {
        ff_loadbalancer::broadcast_task(task);
    }   
};



// it just starts all workers than it exits
class Emitter: public ff_node {
public:
    Emitter(my_loadbalancer * const lb):lb(lb) {}
    void * svc(void * ) {
        std::cerr << "Emitter received task\n";
        lb->broadcast(GO_ON);
        return NULL; 
    }
private:
    my_loadbalancer * lb;
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
    if (argc<5) {
        std::cerr 
            << "use: "  << argv[0] 
            << " ntasks num-integer-x-item #n nticks\n";
        return -1;
    }
    
    unsigned int ntasks         = atoi(argv[1]);
    unsigned int itemsize       = atoi(argv[2]);
    unsigned int nworkers       = atoi(argv[3]);    
    long long    nticks         = strtoll(argv[4],NULL,10);

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
            void * task= MALLOC(itemsize*sizeof(task_t));
            memset(task,0,itemsize*sizeof(task_t));
            FREE(task,itemsize*sizeof(task_t));
        }
        ffTime(STOP_TIME);       
        std::cerr << "DONE, time= " << ffTime(GET_TIME) << " (ms)\n";
        return 0;
    } 

    // create the farm object
    ff_farm<my_loadbalancer> farm;
    // create and add emitter object to the farm
    Emitter E(farm.getlb());
    farm.add_emitter(&E);
    
    std::vector<ff_node *> w;
    for(unsigned int i=0;i<nworkers;++i) 
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
