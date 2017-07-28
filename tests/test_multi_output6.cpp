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
 *     pipe(seq, farm2)           
 *
 *                                 _________________
 *                                |                 |
 *                                |    --> Worker---|--->
 *                                v   |             ^    |
 *                 Seq ---> Emitter--- --> Worker---|---> -->Collector
 *                          ^  ^  ^   |             v    |     |
 *                          |  |  |    --> Worker---|--->      |
 *                          |  |  |_________________|          |
 *                          |  |                               |
 *                          |   -------------------------------
 *           Manager -------
 *
 *
 *
 *
 */

/* Author: Massimo Torquati
 * Date:   July 2017
 *
 */


#include <iostream>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>

using namespace ff;


const int MANAGERID = MAX_NUM_THREADS+100;

typedef enum { ADD, REMOVE } reconf_op_t;

struct Command_t {
    Command_t(int id, reconf_op_t op): id(id), op(op) {}
    int         id;
    reconf_op_t op;
};

struct Seq: ff_node_t<long> {
    long ntasks=0;
    Seq(long ntasks):ntasks(ntasks) {}

    long *svc(long *) {
        for(long i=1;i<=ntasks; ++i)
            ff_send_out((long*)i);
        return EOS;
    }
};


class Emitter: public ff_node_t<long> {
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
    Emitter(ff_loadbalancer *lb):lb(lb) {}

    int svc_init() {
        last = lb->getNWorkers();
        ready.resize(lb->getNWorkers());
        for(size_t i=0; i<ready.size(); ++i) ready[i] = true;
        nready=ready.size();
        return 0;
    }
    long* svc(long* t) {
        int wid = lb->get_channel_id();
        if (wid == -1) { // task coming from seq
            printf("Emitter: TASK FROM INPUT %ld \n", (long)t);
            int victim = selectReadyWorker();
            if (victim < 0) data.push_back(t);
            else {
                lb->ff_send_out_to(t, victim);
                ready[victim]=false;
                --nready;
                onthefly++;
            }
            return GO_ON;
        }
        
        // the id of the manager channel is greater than the maximum id of the workers
        if ((size_t)wid == MANAGERID) {  
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

        if ((size_t)wid < lb->getNWorkers()) { // ack coming from the workers
            printf("Emitter got %ld back from %d data.size=%ld\n", (long)t, wid, data.size());
            assert(ready[wid] == false);
            ready[wid] = true;
            ++nready;
            if (data.size()>0) {
                lb->ff_send_out_to(data.back(), wid);
                onthefly++;
                data.pop_back();
                ready[wid]=false;
                --nready;
            } 
            return GO_ON;
        }
        --onthefly;
        if (eos_received && (nready + sleeping) == ready.size() && onthefly<=0) {
            printf("Emitter exiting\n");
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
            if ((nready + sleeping) == ready.size() && data.size() == 0 && onthefly<=0) {
                printf("EMITTER2 BROADCASTING EOS\n");
                lb->broadcast_task(EOS);
            }
        }
    }
private:
    unsigned eos_received = 0;
    unsigned last, nready, onthefly=0;
    std::vector<bool> ready;
    std::vector<long*> data;
    ff_loadbalancer *lb;    
    int sleeping=0;
};

struct Worker: ff_monode_t<long> {
    int svc_init() {
        printf("Worker id=%ld starting\n", get_my_id());
        return 0;
    }

    long* svc(long* task) {
        printf("Worker id=%ld got %ld\n", get_my_id(), (long)task);
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
        printf("Collector received task = %ld, sending it back to the Emitter\n", (long)(task));
        return task;
    }
};


struct Manager: ff_node_t<Command_t> {
    Manager(): 
        channel(100, true, MANAGERID) {}

    Command_t* svc(Command_t *) {
        struct timespec req = {0, static_cast<long>(5*1000L)};
        nanosleep(&req, NULL);

        Command_t *cmd1 = new Command_t(0, REMOVE);
        channel.ff_send_out(cmd1);

        Command_t *cmd2 = new Command_t(1, REMOVE);
        channel.ff_send_out(cmd2);

        nanosleep(&req, NULL);

        Command_t *cmd3 = new Command_t(1, ADD);
        channel.ff_send_out(cmd3);

        Command_t *cmd4 = new Command_t(0, ADD);
        channel.ff_send_out(cmd4);

        nanosleep(&req, NULL);

        channel.ff_send_out(EOS);

        return GO_OUT;
    }

    void svc_end() {
        printf("Manager ending\n");
    }


    int run() { return ff_node_t<Command_t>::run(); }
    int wait() { return ff_node_t<Command_t>::wait(); }


    ff_buffernode * const getChannel() { return &channel;}

    ff_buffernode  channel;
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
        if (ntasks<500) ntasks = 500;
        if (nworkers <3) nworkers = 3;
    }

    Seq seq(ntasks);

    Manager manager;

    ff_Farm<long>   farm(  [&]() { 
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nworkers;++i)  {
                W.push_back(make_unique<Worker>());
            }
            return W;
        } () );

    // registering the manager channel as one extra input channel for the load balancer
    farm.getlb()->addManagerChannel(manager.getChannel());
    Emitter E(farm.getlb());

    farm.remove_collector();
    farm.add_emitter(E); 
    farm.wrap_around();
    // here the order of instruction is important. The collector must be
    // added after the wrap_around otherwise the feedback channel will be
    // between the Collector and the Emitter
    Collector C;
    farm.add_collector(C);
    farm.wrap_around(true);

    ff_Pipe<> pipe(seq, farm);
    
    manager.run();
    pipe.run_then_freeze();
    pipe.wait_freezing();
    pipe.wait();
    manager.wait();

    return 0;
}
