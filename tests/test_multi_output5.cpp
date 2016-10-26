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
 *     pipe(farm1, pipe_internal(farm2, Collector) )   version 1
 *     pipe(farm1, farm2)                              version 2
 *
 *        | ----------------------------- pipe --------------------------------------- |                                      
 *                                              | ------------- pipe_internal -------- |
 *                      farm1                         farm2                   Collector
 *                                               ____________________
 *                                              |                    |
 *                     --> Worker1 -->          |       --> Worker2--|--->
 *                    |               |         v      |             ^    |
 *         Emitter1 -- --> Worker1 -->  --> Emitter2 -- --> Worker2--|---> -->Collector                  
 *                    |               |         ^      |             v    |
 *                     --> Worker1 -->          |       --> Worker2--|--->
 *                                              |____________________|   
 *                                 
 *                                 
 *  The first farm does not have the collector stage. 
 *  Emitter2 implements a "real" on-demand scheduling policy (it sends the task 
 *  only to the workers that are ready to compute). 
 *  In version1 the Collector stage is not part of the farm2 but is a multi-input node
 *  that is part of the pipe_internal. In version2 the Collector is part of the farm2 stage.
 */

#include <iostream>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>

using namespace ff;

struct Emitter1: public ff_node_t<long> {
    Emitter1(long ntasks):ntasks(ntasks) {}
    long* svc(long*) {
        for(long i=0;i<ntasks;++i)
            ff_send_out((long*)(i+10));
        return EOS;
    }
    long ntasks;
};
struct Worker1: ff_node_t<long> {
    long* svc(long* task) { return task;  }
};


class Emitter2: public ff_node_t<long> {
protected:
    int selectReadyWorker() {
        for (unsigned i=last+1;i<ready.size();++i) {
            if (ready[i]) {
                last = i;
                return i;
            }
        }
        for (unsigned i=0;i<=last;++i) {
            if (ready[i]) {
                last = i;
                return i;
            }
        }
        return -1;
    }
public:
    Emitter2(ff_loadbalancer *lb):lb(lb) {}

    int svc_init() {
        last = lb->getNWorkers();
        ready.resize(lb->getNWorkers());
        for(size_t i=0; i<ready.size(); ++i) ready[i] = true;
        nready=ready.size();
        return 0;
    }
    long* svc(long* t) {
        int wid = lb->get_channel_id();
        if (wid == -1) { // task coming from the first farm
            printf("TASK FROM INPUT %ld \n", (long)t);
            int victim = selectReadyWorker();
            if (victim < 0) data.push_back(t);
            else {
                lb->ff_send_out_to(t, victim);
                ready[victim]=false;
                --nready;
            }
            return GO_ON;
        }
        printf("Emitter2 got %ld back from %d data.size=%ld\n", (long)t, wid, data.size());
        assert(ready[wid] == false);
        ready[wid] = true;
        ++nready;
        if (data.size()>0) {
            lb->ff_send_out_to(data.back(), wid);
            data.pop_back();
            ready[wid]=false;
            --nready;
        } else  if (eos_received==ready.size() && nready == ready.size()) {
            printf("Emitter2 exiting\n");
            return EOS;
        }
        return GO_ON;
    }
    void svc_end() {
        // just for debugging
        assert(data.size()==0);
    }
    void eosnotify(ssize_t id) {
        if (id == -1) { // we have to receive all EOS from the previous stage
            eos_received++; 
            printf("EOS received eos_received = %u nready = %u\n", eos_received, nready);
            if (eos_received == ready.size() && 
                nready == ready.size() && data.size() == 0) {
                printf("EMITTER2 BROADCASTING EOS\n");
                lb->broadcast_task(EOS);
            }
        }
    }
private:
    unsigned eos_received = 0;
    unsigned last, nready;
    std::vector<bool> ready;
    std::vector<long*> data;
    ff_loadbalancer *lb;    
};

struct Worker2: ff_monode_t<long> {
    long* svc(long* task) {
        ff_send_out_to(task, 1);  // to the next stage 
        ff_send_out_to(task, 0);  // send the "ready msg" to the emitter 
        return GO_ON;
    }
};

// multi-input stage
struct Collector: ff_minode_t<long> {
    long* svc(long* task) {
        printf("Collector received task = %ld\n", (long)(task));
        return GO_ON;
    }
};

int main(int argc, char* argv[]) {
    unsigned nworkers = 3;
    int ntasks = 1000;
    if (argc>1) {
        if (argc < 3) {
            std::cerr << "use:\n" << " " << argv[0] << " numworkers ntasks\n";
            return -1;
        }
        nworkers  =atoi(argv[1]);
        ntasks    =atoi(argv[2]);
    }

    ff_Farm<long>   farm1(  [&]() { 
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nworkers;++i)  {
                W.push_back(make_unique<Worker1>());
            }
            return W;
        } () );
    Emitter1 E1(ntasks);
    farm1.remove_collector();
    farm1.add_emitter(E1);  

    ff_Farm<long>   farm2(  [&]() { 
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nworkers;++i)  {
                W.push_back(make_unique<Worker2>());
            }
            return W;
        } () );
    Emitter2 E2(farm2.getlb());
    farm2.remove_collector();
    farm2.add_emitter(E2); 
#if 0   // version 1
    // the farm2 is a multi-input pattern, if setMultiInput call appers 
    // after the wrap_around call, then we have to call wrap_around(true)
    farm2.setMultiInput(); 
    farm2.wrap_around(); 
    Collector C;
    ff_Pipe<long> pipe_internal(farm2, C);

    ff_Pipe<> pipe(farm1, pipe_internal);

#else   // version 2

    farm2.setMultiInput();
    farm2.wrap_around();
    // here the order of instruction is important. The collector must be
    // added after the wrap_around otherwise the feedback channel will be
    // between the Collector and the Emitter2
    Collector C;
    farm2.add_collector(C);

    ff_Pipe<> pipe(farm1, farm2);
#endif
    pipe.run_and_wait_end();
    return 0;
}
