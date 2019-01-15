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
 * Simple Farm without collector. Tasks are allocated dinamically by the 
 * ff_allocator and all tasks have a fixed size (itemsize*sizeof(ff_task_t)).
 *
 */

#include <sys/types.h>
//#include <unistd.h>
#include <stdlib.h>

#include <vector>
#include <iostream>
#include <ff/ff.hpp>
#include <ff/allocator.hpp>
#include <ff/cycle.h>

using namespace ff;

typedef int ff_task_t;

static ff_allocator ffalloc;

// generic worker
class Worker: public ff_node {
protected:
    void do_work(int * task, int size, long long nticks) {
        for(int i=0;i<size;++i)
            task[i]+=1;
        
        ticks_wait(nticks);
    }


public:
    Worker(int itemsize, long long nticks):itemsize(itemsize),nticks(nticks) {}

    // called just one time at the very beginning
    int svc_init() {
        if (ffalloc.register4free()<0) {
            error("Worker, register4free fails\n");
            return -1;
        }
        return 0;
    }

    void * svc(void * t) {
        ff_task_t * task = (ff_task_t *)t;

        do_work(task,itemsize,nticks);

        ffalloc.free(task);
        // we don't have the collector so we have any task to send out
        return GO_ON; 
    }

private:
    int itemsize;
    long long nticks;
};


// the load-balancer filter
class Emitter: public ff_node {
protected:

    inline void filltask(int * task, size_t size) {
        static int val = 0;
        for(unsigned int i=0;i<size;++i)
            task[i]=val++;
    }

public:
    Emitter(int max_task, int itemsize):ntask(max_task),itemsize(itemsize) {
        ffalloc.init();
    };

    // called just one time at the very beginning
    int svc_init() {
        if (ffalloc.registerAllocator()<0) {
            error("Emitter, registerAllocator fails\n");
            return -1;
        }
        return 0;
    }

    void * svc(void *) {
        static int n=0;
        ff_task_t * task = (ff_task_t*)ffalloc.malloc(itemsize*sizeof(ff_task_t));
        if (!task) abort();
        filltask(&task[1], itemsize-1);
        task[0] = n++;
        --ntask;
        if (ntask<0) return NULL;
        return task;
    }

    // I don't need the following 
    //void  svc_end()  {}
private:
    int ntask;
    int itemsize;
};


int main(int argc, char * argv[]) {    
    unsigned int buffer_entries = 8192;
    unsigned int streamlen      = 1000000;
    unsigned int itemsize       = 16;
    unsigned int nworkers       = 3;
    long long nticks			= 1000;
    
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
        nticks			= strtoll(argv[5],NULL,10);
    }

    // arguments check
    if (!nworkers || !streamlen || nticks<0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    // create the farm object
    ff_farm farm(false, buffer_entries);
    std::vector<ff_node *> w;
    for(unsigned int i=0;i<nworkers;++i) 
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
    return 0;
}
