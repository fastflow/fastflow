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
 *       2-stages pipeline
 *               
 *                       ------------------- <--
 *                      |                   |   |
 *                      |        ---> FU ---    |
 *                      v       |               |
 *        MU ----> Scheduler --- ---> FU ------- 
 *                      ^       |
 *                      |        ---> FU ---
 *                      |                   |
 *                       -------------------
 *
 *  Example of macro-dataflow pattern with input throttling in the Scheduler node.
 *
 */

#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>
#include <ff/node.hpp>
#include <ff/allocator.hpp>

using namespace ff;

struct task_t {
    task_t(int op, long t):op(op),t(t) {}
    int op;
    long t;
};


class MU: public ff_node {
protected:
    inline void* MALLOC(size_t size) {
        return ffalloc->malloc(size);
    }
public:
    MU(int numtasks, ff_allocator* ffalloc):
        numtasks(numtasks),ffalloc(ffalloc) {}

    int svc_init() {
        if (!ffalloc) return -1;
        if (ffalloc->registerAllocator()<0) {
            error("MU, registerAllocator fails\n");
            return -1;
        }
        return 0;
    }

    void* svc(void*) {       
        task_t* t;
        for(long i=1;i<=numtasks;++i) {
            
            t = new task_t(1,i%10);
            t = new (MALLOC(sizeof(task_t))) task_t(1,i%10);
            
            while (!ff_send_out(t, 1)) ff_relax(1);
        }
        t = new task_t(0,-1);
        while (!ff_send_out(t, 1)) ff_relax(1);
        return NULL;
    }
private:
    long numtasks;
    ff_allocator* ffalloc;
};

/*
 * NOTE: This is NOT a multi-input node !
 */
class Scheduler: public ff_node {
protected:
    void FREE(void* ptr) {
        ffalloc->free(ptr);
    }
public:
    Scheduler(ff_loadbalancer* lb, ff_allocator* ffalloc):
        lb(lb),ffalloc(ffalloc),wait(0),cnt(0),numtask(0),finished(false) {}

    int svc_init() {
        if (!ffalloc) return -1;
        if (ffalloc->register4free()<0) {
            error("MU, register4free fails\n");
            return -1;
        }
        return 0;
    }
    void* svc(void* task) {
        if (task==NULL) { // if input is disabled svc is called with NULL
            if (finished) {
                assert(numtask!=0);
                return GO_ON;
            }
            // wait a while before restarting reading input
            if (++wait >= 5) {
                wait = 0;
                ff_node::input_active(true);
                printf("STOP WAITING\n");
            }
            return GO_ON;
        }
        // this is a real task
        task_t* t = (task_t*)task;
        if (lb->get_channel_id() == -1) {  // ... received from input channel
            if (t->op == 0) { // is this the end ?
                finished=true;
                ff_node::input_active(false); // we don't want to read FF_EOS
                printf("ENDING PHASE\n");
                if (numtask == 0) { // is it really the end ?
                    printf("END\n");
                    return NULL;
                }
                return GO_ON;
            } 
            ++cnt; ++numtask;
            if (cnt >= 5) { // after 5 inputs ...
                // ... we deactiate input for a while
                cnt=0;
                ff_node::input_active(false);
            }
            return task;
        }
        --numtask;
        assert(numtask>=0);
        FREE(task);
        if (numtask==0 && finished) {
            printf("END\n");
            return NULL;
        }
        return GO_ON;
    }
protected:
    ff_loadbalancer* lb;
    ff_allocator* ffalloc;
    int wait;
    int cnt;
    int numtask;
    bool finished;
};


class FU: public ff_node {
public:
    void* svc(void* task) {
        printf("FU (%d) got one task\n", get_my_id());
        task_t* t = (task_t*)task;
        assert(t->op != 0);
        switch(t->op) {
        case 1: usleep(10); break;
        case 2: usleep(100); break;
        case 3: usleep(1000); break;
        default: usleep(200);
        }
        return task;
    }
};



int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "use:\n" << " " << argv[0] << " numworkers ntasks\n";
        return -1;
    }
    int nw=atoi(argv[1]);
    int numtasks=atoi(argv[2]); 

    // prepare the instruction allocator
    ff_allocator* ffalloc=new ff_allocator();
    int slab = ffalloc->getslabs(sizeof(task_t));
    int nslabs[N_SLABBUFFER];
    if (slab<0) {                               
        if (ffalloc->init()<0) abort();         
    } else {                                
        for(int i=0;i<N_SLABBUFFER;++i) {     
            if (i==slab) nslabs[i]=8192;      
            else nslabs[i]=0;           
        }                                     
        if (ffalloc->init(nslabs)<0) abort(); 
    }                                       
    

    ff_pipeline pipe(false, 10);  // queues in the pipeline are bounded !
    ff_farm<>   farm;
    std::vector<ff_node *> w;
    for(int i=0;i<nw;++i) 
        w.push_back(new FU);
    farm.add_emitter(new Scheduler(farm.getlb(),ffalloc));
    farm.add_workers(w);
    /* -------------------- */
    farm.wrap_around();
    /* -------------------- */
    pipe.add_stage(new MU(numtasks,ffalloc));
    pipe.add_stage(&farm);


    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    return 0;
}
