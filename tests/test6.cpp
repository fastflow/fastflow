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
 * This program tests composition of farm modules.
 *
 *         |farm(worker1)|
 *         |
 *    farm |farm(worker2)|
 *         |
 *         |farm(worker3)|
 *
 */

#include <iostream>
#include <vector>
#include <ff/ff.hpp>
#include <ff/allocator.hpp>


using namespace ff;

static ff_allocator ffalloc;
enum { MIN_TASK_SIZE=32, MAX_TASK_SIZE=16384 };


class Worker1: public ff_node {
public:
    void * svc(void * task) {
        std::cout << "Worker1 id= " << get_my_id() << " got task " << *(int*)task << "\n";
        return task; 
    }
};

class Worker2: public ff_node {
public:
    void * svc(void * task) {
        std::cout << "Worker2 id= " << get_my_id() << " got task " << *(int*)task << "\n";
        return task; 
    }
};

class Worker3: public ff_node {
public:
    void * svc(void * task) {
        std::cout << "Worker3 id= " << get_my_id() << " got task " << *(int*)task << "\n";
        return task; 
    }
};


// the load-balancer filter
class Emitter: public ff_node {
public:
    Emitter(int max_task):ntask(max_task) {
        srandom(::getpid()+(getusec()%4999));
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
        size_t size = random() % MAX_TASK_SIZE;
        if (!size) size=MIN_TASK_SIZE;
        int * task = (int*)ffalloc.malloc(size);
        if (!task) abort();
        task[0] = --ntask;
        if (ntask<0) return NULL;
        return task;
    }
private:
    int ntask;
};


// the gatherer filter
class Collector: public ff_node {
public:
    int svc_init() {
        if (ffalloc.register4free()<0) {
            error("Collector, register4free fails\n");
            return -1;
        }
        return 0;
    }
    void * svc(void * task) {
        std::cout << "Collector got task " << *(int*)task << "\n";
        ffalloc.free(task);
        return task;
    }
};



int main(int argc, char * argv[]) {
    int streamlen = 1000;
    if (argc>1) {
        if (argc!=2) {
            std::cerr << "use: "  << argv[0] << " streamlen\n";
            return -1;
        }
        streamlen=atoi(argv[1]);
    }

    // init allocator
    ffalloc.init();

    // bild main farm
    ff_farm farm;

    Emitter e(streamlen);
    Collector c;
    farm.add_emitter(&e);
    farm.add_collector(&c);
    

    ff_farm farm1, farm2, farm3;
    farm1.add_collector(NULL); 
    farm2.add_collector(NULL); 
    farm3.add_collector(NULL); 

    std::vector<ff_node *> w;
    w.push_back(new Worker1);
    w.push_back(new Worker1);
    farm1.add_workers(w);

    w.clear();

    w.push_back(new Worker2);
    w.push_back(new Worker2);
    w.push_back(new Worker2);
    farm2.add_workers(w);

    w.clear();

    w.push_back(new Worker3);
    farm3.add_workers(w);

    w.clear();

    w.push_back(&farm1);
    w.push_back(&farm2);
    w.push_back(&farm3);
    farm.add_workers(w);
    
    if (farm.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    return 0;
}
