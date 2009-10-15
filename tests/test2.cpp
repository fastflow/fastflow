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
 * Simple Farm without collector. Tasks are allocated dinamically using 
 * ff_allocator. 
 *
 */

#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include <iostream>
#include <farm.hpp>
#include <allocator.hpp>

using namespace ff;

typedef int task_t;

static ff_allocator ffalloc;

enum { MIN_TASK_SIZE=32, MAX_TASK_SIZE=16384 };


// generic worker
class Worker: public ff_node {
public:
    // called just one time at the very beginning
    int svc_init(void *) {
        std::cout << "Worker << " << pthread_self() << " svc_init called\n";
        if (ffalloc.register4free()<0) {
            error("Worker, register4free fails\n");
            return -1;
        }
        return 0;
    }

    void * svc(void * task) {
        ffalloc.free(task);
        std::cout << "Worker " << get_my_id() << " freed task\n";
        return NULL; // we don't have the collector so any task to send out
    }
    // I don't need the following 
    //void  svc_end(void * result)  {}
};


// the load-balancer filter
class Emitter: public ff_node {
public:
    Emitter(int max_task):ntask(max_task) {
        srandom(::getpid()+(getusec()%4999));
        ffalloc.init();
    };

    // called just one time at the very beginning
    int svc_init() {
        std::cout << "Emitter svc_init called\n";
        if (ffalloc.registerAllocator()<0) {
            error("Emitter, registerAllocator fails\n");
            return -1;
        }
        return 0;
    }

    void * svc(void *) {
        size_t size = random() % MAX_TASK_SIZE;
        if (!size) size=MIN_TASK_SIZE;
        task_t * task = (task_t*)ffalloc.malloc(size);
        if (!task) abort();
        std::cout << "Emitter allocated task size= " << size << "\n";

        --ntask;
        if (ntask<0) return NULL;
        return task;
    }

    // I don't need the following 
    //void  svc_end()  {}
private:
    int ntask;
};


int main(int argc, 
         char * argv[]) {
    
    if (argc<3) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " nworkers streamlen\n";
        return -1;
    }
    
    int nworkers=atoi(argv[1]);
    int streamlen=atoi(argv[2]);
    
    if (!nworkers || !streamlen) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    ff_farm<Worker> farm(nworkers);
    
    Emitter E(streamlen);
    farm.add_emitter(E);
    
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    
    std::cerr << "DONE, time= " << farmTime(GET_TIME) << " (ms)\n";
    return 0;
}
