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
 * This test shows how to implement a complete user level scheduling policy,
 * using memory to 
 *
 */
#include <vector>
#include <iostream>
#include <ff/ff.hpp>
#include <ff/allocator.hpp>
#include <ff/utils.hpp>  

using namespace ff;

#define MAX_TIME_MS  100000


// generic worker
class Worker: public ff_node {
public:
    Worker():ntasks(0) {}

    void * svc(void * task) {
        ++ntasks;
        ticks_wait(random()%MAX_TIME_MS);
        return GO_ON;
    }     
    void svc_end() {
        printf("Worker %ld gets %ld tasks\n", get_my_id(), ntasks);
    }
private:
    long ntasks;
};

class my_loadbalancer: public ff_loadbalancer {
protected:
    // called after that the worker has been selected
    // can be used to allocate/reuse memory for the task
    // once it is known who is the 'victim'
    inline void* callback(int n, void *task) { 
        assert((size_t)n<ff_loadbalancer::getnworkers());
        if (pool[n]==NULL) pool[n] = (void*)malloc(128*sizeof(char));
        else printf("ALREADY allocated\n");
        victim=n;
        return task;
    }
public:
    my_loadbalancer(int max_num_workers):ff_loadbalancer(max_num_workers),poolsize(max_num_workers) {
        pool=(void**)malloc(poolsize*sizeof(void*));
        assert(pool != NULL);
        for(int i=0;i<poolsize;++i) pool[i]=NULL;
        victim=-1;
    }

    void putDone() { 
        assert(victim>-1 && pool[victim]!=NULL); 
        pool[victim]=NULL; 
        victim=-1; 
    } 
private:
    int    victim;
    int    poolsize;
    void** pool;
};

// emitter filter
class Emitter: public ff_node {
public:
    Emitter(size_t ntasks, my_loadbalancer * const lb):
        ntasks(ntasks),lb(lb) {}

    void * svc(void *) {
        for(size_t i=1;i<=ntasks;++i) {
            ff_send_out((void*)i);
            lb->putDone();
        }
        return EOS;
    }
    
private:
    size_t ntasks;
    my_loadbalancer * lb;
};



int main(int argc, char * argv[]) {    
    int    nworkers = 4;
    size_t ntask = 1000;
    if (argc>1) {
        if (argc<3) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nworkers ntask\n";
            return -1;
        }
        
        nworkers=atoi(argv[1]);
        ntask=atol(argv[2]);
    }
    if (nworkers<=0 || ntask<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    ff_farm<my_loadbalancer> farm;
    Emitter emitter(ntask,farm.getlb());
    farm.add_emitter(&emitter);
    farm.set_scheduling_ondemand();

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);

    farm.run_and_wait_end();

    std::cerr << "DONE\n";
    return 0;
}
