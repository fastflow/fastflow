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
/* testing BLK and NBLK */

/* Author: Massimo Torquati
 *  pipe(First, loopback(farm(Scheduler, Worker))) no collector
 */


#include <vector>
#include <iostream>
#include <ff/ff.hpp>
#include <ff/allocator.hpp>

using namespace ff;

struct fftask_t {
    fftask_t(int op, long t):op(op),t(t) {}
    int op;
    long t;
};

class First: public ff_node_t<fftask_t> {
protected:
    inline void* MALLOC(size_t size) {
        return ffalloc->malloc(size);
    }
public:
    First(int numtasks, ff_allocator* ffalloc):
        numtasks(numtasks),ffalloc(ffalloc) {}

    int svc_init() {
        if (!ffalloc) return -1;
        if (ffalloc->registerAllocator()<0) {
            error("MU, registerAllocator fails\n");
            return -1;
        }
        return 0;
    }
    
    fftask_t* svc(fftask_t*) {       
        fftask_t* t;

        for(size_t i=1;i<=numtasks;++i) {
            
            t = new fftask_t(1,i%10);
            t = new (MALLOC(sizeof(fftask_t))) fftask_t(1,i%10);

            switch(i) {
            case 100:{
                while(!ff_send_out(BLK));
            } break;
            case 500: {
                while(!ff_send_out(NBLK));
            } break;
            case 800: {
                while(!ff_send_out(BLK));
            } break;
            }

            while (!ff_send_out(t, 1)) ff_relax(1);
        }
        printf("LB: GENERATO EOS\n");
        return EOS;
    }

private:
    size_t numtasks;
    ff_allocator* ffalloc;
};

class Scheduler: public ff_node_t<fftask_t> {
protected:
    void FREE(void* ptr) {
        ffalloc->free(ptr);
    }
public:
    Scheduler(ff_loadbalancer* lb, ff_allocator* ffalloc):
        lb(lb),ffalloc(ffalloc),numtask(0) {}

    int svc_init() {
        if (!ffalloc) return -1;
        if (ffalloc->register4free()<0) {
            error("MU, register4free fails\n");
            return -1;
        }
        return 0;
    }
    fftask_t* svc(fftask_t* task) {
        if (lb->get_channel_id() == -1) { 
            ++numtask;
            return task;
        }
        --numtask;
        assert(numtask>=0);
        FREE(task);
        if (numtask==0 && eos_arrived) {
            return EOS;
        }
        return GO_ON;
    }

    void eosnotify(ssize_t id) {
        if (id==-1) {
            printf("LB got EOS\n");
            eos_arrived = true;
        }
    }

protected:
    ff_loadbalancer* lb;
    ff_allocator* ffalloc;
    long numtask;
    bool eos_arrived=false;
};


class Worker: public ff_node_t<fftask_t> {
public:
    fftask_t* svc(fftask_t* task) {
        fftask_t* t = (fftask_t*)task;
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
    size_t nw=3;
    size_t numtasks=1000;
    if (argc>1) {
        if (argc < 3) {
            std::cerr << "use:\n" << " " << argv[0] << " numworkers ntasks\n";
            return -1;
        }
        nw=atol(argv[1]);
        numtasks=atoi(argv[2]); 
    }
    // prepare the instruction allocator
    ff_allocator* ffalloc=new ff_allocator();
    int slab = ffalloc->getslabs(sizeof(fftask_t));
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
    
    ff_Farm<fftask_t>   farm(  [nw]() { 
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nw;++i)  W.push_back(make_unique<Worker>());
            return W;
        } () );
    Scheduler sched(farm.getlb(), ffalloc);
    farm.add_emitter(sched);
    farm.remove_collector();
    farm.setFixedSize(true);
    farm.setInputQueueLength(nw*100);
    farm.setOutputQueueLength(nw*100);
    farm.wrap_around();
    
    First first(numtasks, ffalloc);
    ff_Pipe<> pipe(first, farm);
    pipe.setXNodeInputQueueLength(100,true);
    pipe.setXNodeOutputQueueLength(100,true);
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    return 0;
}
