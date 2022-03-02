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
 *                                          | -------------- pipe_internal ----------- |
 *                      farm1                         farm2                   Collector
 *                                               ____________________
 *                                              |                    |
 *                     --> Worker1 -->          |       --> Worker2--|--->
 *                    |               |         v      |             ^    |
 *         Emitter1 -- --> Worker1 -->  --> Emitter2 -- --> Worker2--|---> -->Collector
 *          ^         |               |     ^   ^      |             v    |
 *          |          --> Worker1 -->      |   |       --> Worker2--|--->
 * channel1 |                               |   |____________________|   
 *          |                               |
 *          |         channel2              |
 *       Manager ---------------------------
 *
 *
 *                                 
 *  The first farm does not have the collector stage. 
 *  Emitter2 implements a "real" on-demand scheduling policy (it sends the task 
 *  only to the workers that are ready to compute). 
 *  In version1 the Collector stage is not part of the farm2 but is a multi-input node
 *  that is part of the pipe_internal. In version2 the Collector is part of the farm2 stage.
 *
 *  The Manager is "manually" added to the pipeline. It sends reconfiguration commands 
 *  (ADD, REMOVE workers) to the two schedulers.
 *
 */

/* Author: Massimo Torquati
 *
 *
 */


#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

typedef enum { ADD, REMOVE } reconf_op_t;

struct Command_t {
    Command_t(int id, reconf_op_t op): id(id), op(op) {}
    int         id;
    reconf_op_t op;
};


struct Emitter1: public ff_node_t<long> {
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
        abort();
        return -1;
    }
public:
    Emitter1(long ntasks, ff_loadbalancer *lb, ff_buffernode *const cmdChannel):ntasks(ntasks), lb(lb), cmdChannel(cmdChannel) {}

    int svc_init() {
        last = lb->getNWorkers();
        ready.resize(lb->getNWorkers());
        for(size_t i=0; i<ready.size(); ++i) ready[i] = true;        
        return 0;
    }

    long* svc(long*) {
        bool manager_alive = true;
        for(long i=0;i<ntasks;++i) {
            // from time to time the scheduler checks if there are 
            // commands in the command channel 
            if (i >= 10 && ((i % 10) == 0) && manager_alive) {
                Command_t *cmd = nullptr;
                cmdChannel->gather_task(cmd, 1);
                if (cmd == (Command_t*)EOS) { 
                    manager_alive = false;
                    continue;
                }
                if (cmd) {
                    printf("EMITTER1 (i=%ld), SENDING %s to WORKER %d\n", i, cmd->op==ADD?"ADD":"REMOVE", cmd->id);

                    switch(cmd->op) {
                    case ADD:     {
                        lb->thaw(cmd->id, true);
                        ready[cmd->id] = true;
                    } break;
                    case REMOVE:  {
                        lb->ff_send_out_to(GO_OUT, cmd->id);
                        ready[cmd->id] = false;
                    } break;
                    default: abort();
                    }
                    delete cmd;
                }
            }
            lb->ff_send_out_to((long*)(i+10), selectReadyWorker());
        }
        return EOS;
    }

    void svc_end() {
        for(size_t i=0; i<ready.size(); ++i)
            if (!ready[i]) lb->thaw(i, false);

    }

    
    long ntasks;
    ff_loadbalancer *lb;
    unsigned last;
    std::vector<bool> ready;

    ff_buffernode *const cmdChannel = nullptr;
};
struct Worker1: ff_node_t<long> {
    int svc_init() {
        printf("Worker1 id=%ld starting\n", get_my_id());
        return 0;
    }
    long* svc(long* task) { 
        printf("Worker1 id=%ld got %ld\n", get_my_id(), (long)task);
        return task;  
    }
    void svc_end() {
        printf("Worker1 id=%ld going to sleep\n", get_my_id());
    }
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
            printf("Emitter2: TASK FROM INPUT %ld \n", (long)t);
            int victim = selectReadyWorker();
            if (victim < 0) data.push_back(t);
            else {
                lb->ff_send_out_to(t, victim);
                ready[victim]=false;
                --nready;
            }
            return GO_ON;
        }
        
        // the id of the manager channel is greater than the maximum id of the workers
        if ((size_t)wid > lb->getNWorkers()) {  
            Command_t *cmd = reinterpret_cast<Command_t*>(t);
            printf("EMITTER2 SENDING %s to WORKER %d\n", cmd->op==ADD?"ADD":"REMOVE", cmd->id);
            switch(cmd->op) {
            case ADD:     lb->thaw(cmd->id, true);             break;
            case REMOVE:  lb->ff_send_out_to(GO_OUT, cmd->id); break;
            default: abort();
            }
            delete cmd;            
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
        } else  if (eos_received ==ready.size() && (nready + sleeping) == ready.size()) {
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
            if (eos_received  == ready.size() && 
                (nready + sleeping) == ready.size() && data.size() == 0) {
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
    int sleeping=0;
};

struct Worker2: ff_monode_t<long> {
    int svc_init() {
        printf("Worker2 id=%ld starting\n", get_my_id());
        return 0;
    }

    long* svc(long* task) {
        printf("Worker2 id=%ld got %ld\n", get_my_id(), (long)task);
        ff_send_out_to(task, 1);  // to the next stage 
        ff_send_out_to(task, 0);  // send the "ready msg" to the emitter 
        return GO_ON;
    }

    void svc_end() {
        printf("Worker2 id=%ld going to sleep\n", get_my_id());
    }

};

// multi-input stage
struct Collector: ff_minode_t<long> {
    long* svc(long* task) {
        printf("Collector received task = %ld\n", (long)(task));
        return GO_ON;
    }
};


struct Manager: ff_node_t<Command_t> {

    Manager(): 
        channel1(100, true, MAX_NUM_THREADS+100),  
        channel2(100, true, MAX_NUM_THREADS+200) {

        // The following is needed if the program runs in BLOCKING_MODE
        // In this case, channel1 and channel2 will not be automatically
        // initialized so we have to do it manually. Moreover, even if we
        // want to run in BLOCKING_MODE, channel1 and channel2 cannot be
        // configured in blocking mode for the output, that is, ff_send_out
        // has to be non-blocking!!!!
        channel1.init_blocking_stuff();
        channel2.init_blocking_stuff();        
        channel1.reset_blocking_out();
        channel2.reset_blocking_out();        
    }

    Command_t* svc(Command_t *) {
        
        Command_t *cmd1 = new Command_t(0, REMOVE);
        channel1.ff_send_out(cmd1);
        Command_t *cmd2 = new Command_t(1, REMOVE);
        channel2.ff_send_out(cmd2);

        Command_t *cmd3 = new Command_t(1, REMOVE);
        channel1.ff_send_out(cmd3);
        Command_t *cmd4 = new Command_t(0, REMOVE);
        channel2.ff_send_out(cmd4);

        Command_t *cmd5 = new Command_t(1, ADD);
        channel1.ff_send_out(cmd5);
        Command_t *cmd6 = new Command_t(1, ADD);
        channel2.ff_send_out(cmd6);

        Command_t *cmd7 = new Command_t(0, ADD);
        channel1.ff_send_out(cmd7);
        Command_t *cmd8 = new Command_t(0, ADD);
        channel2.ff_send_out(cmd8);

        channel1.ff_send_out(EOS);
        channel2.ff_send_out(EOS);

        return GO_OUT;
    }

    void svc_end() {
        printf("Manager ending\n");
    }


    int run(bool=false) {
        return ff_node_t<Command_t>::run();
    }
    int wait() { return ff_node_t<Command_t>::wait(); }


    ff_buffernode * getChannel1() { return &channel1;}
    ff_buffernode * getChannel2() { return &channel2;}


    ff_buffernode  channel1;
    ff_buffernode  channel2;
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
        if (ntasks<100) ntasks = 100;
        if (nworkers <3) nworkers = 3;
    }

    Manager manager;

    ff_Farm<long>   farm1(  [&]() { 
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nworkers;++i)  {
                W.push_back(make_unique<Worker1>());
            }
            return W;
        } () );
    Emitter1 E1(ntasks, farm1.getlb(), manager.getChannel1());
    farm1.remove_collector();
    farm1.add_emitter(E1);  

    ff_Farm<long>   farm2(  [&]() { 
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nworkers;++i)  {
                W.push_back(make_unique<Worker2>());
            }
            return W;
        } () );


    // registering the manager channel as one extra input channel for the load balancer
    farm2.getlb()->addManagerChannel(manager.getChannel2());
    Emitter2 E2(farm2.getlb());
    
    farm2.remove_collector();
    farm2.add_emitter(E2); 
#if 0   // version 1
    farm2.wrap_around(); 
    Collector C;
    ff_Pipe<long> pipe_internal(farm2, C);

    ff_Pipe<> pipe(farm1, pipe_internal);
#else   // version 2
    //farm2.setMultiInput();  // not needed anymore!
    farm2.wrap_around();
    // here the order of instruction is important. The collector must be
    // added after the wrap_around otherwise the feedback channel will be
    // between the Collector and the Emitter2
    Collector C;
    farm2.add_collector(C);

    ff_Pipe<> pipe(farm1, farm2);
    
#endif

    pipe.run_then_freeze();
    manager.run();
    pipe.wait_freezing();
    pipe.wait();
    manager.wait();

    return 0;
}
