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
 *  NOTE: the Emitter can be a standard ff_node_t and in this case it has to use the 
 *        ff_loadbalancer of the farm (to test this case define EMITTER_FF_NODE. 
 *        Alternatively, it can be a ff_monode_t and in this case its internal load-balancer
 *        will be (automatically) attached to that of the farm.
 *
 */

/* Author: Massimo Torquati
 * Date:   July 2017
 *
 */


#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

// this is an id greater than all ids
const int MANAGERID = MAX_NUM_THREADS+100;


typedef enum { ADD, REMOVE } reconf_op_t;
struct Command_t {
    Command_t(int id, reconf_op_t op): id(id), op(op) {}
    int         id;
    reconf_op_t op;
};

// first stage
struct Seq: ff_node_t<long> {
    long ntasks=0;
    Seq(long ntasks):ntasks(ntasks) {}

    long *svc(long *) {
        for(long i=1;i<=ntasks; ++i) {
            ff_send_out((long*)i);

            struct timespec req = {0, static_cast<long>(5*1000L)};
            nanosleep(&req, NULL);
        }
        return EOS;
    }
};

// scheduler 
class Emitter: public ff_monode_t<long> {
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
    int svc_init() {
        last = get_num_outchannels();  // at the beginning, this is the number of workers
        ready.resize(last); 
        sleeping.resize(last);
        for(size_t i=0; i<ready.size(); ++i) {
            ready[i]    = true;
            sleeping[i] = false;
        }
        nready=ready.size();
        return 0;
    }
    long* svc(long* t) {       
        int wid = get_channel_id();
        
        // the id of the manager channel is greater than the maximum id of the workers
        if ((size_t)wid == MANAGERID) {  
            Command_t *cmd = reinterpret_cast<Command_t*>(t);
            printf("EMITTER2 SENDING %s to WORKER %d\n", cmd->op==ADD?"ADD":"REMOVE", cmd->id);
            switch(cmd->op) {
            case ADD:     {
                ff_monode::getlb()->thaw(cmd->id, true);
                assert(sleeping[cmd->id]);
                sleeping[cmd->id] = false;
                frozen--;
            } break;
            case REMOVE:  {
                ff_send_out_to(GO_OUT, cmd->id);
                assert(!sleeping[cmd->id]);
                sleeping[cmd->id] = true;
                frozen++;
            } break;
            default: abort();
            }
            delete cmd;            
            return GO_ON;
        }

        if (wid == -1) { // task coming from seq
            //printf("Emitter: TASK FROM INPUT %ld \n", (long)t);
            int victim = selectReadyWorker();
            if (victim < 0) data.push_back(t);
            else {
                ff_send_out_to(t, victim);
                ready[victim]=false;
                --nready;
                onthefly++;
            }
            return GO_ON;
        }
        
        if ((size_t)wid < get_num_outchannels()) { // ack coming from the workers
            //printf("Emitter got %ld back from %d data.size=%ld, onthefly=%d\n", (long)t, wid, data.size(), onthefly);
            assert(ready[wid] == false);
            ready[wid] = true;
            ++nready;
            if (data.size()>0) {
                ff_send_out_to(data.back(), wid);
                onthefly++;
                data.pop_back();
                ready[wid]=false;
                --nready;
            } 
            return GO_ON;
        }

        // task coming from the Collector
        --onthefly;
        //printf("Emitter got %ld back from COLLECTOR data.size=%ld, onthefly=%d\n", (long)t, data.size(), onthefly);
        if (eos_received && ((nready + frozen) == ready.size()) && (onthefly<=0)) {
            printf("Emitter EXITING\n");
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
            eos_received = true;
            //printf("EOS received eos_received = %u nready = %u\n", eos_received, nready);
            if (((nready+frozen) == ready.size()) && (data.size() == 0) && onthefly<=0) {

                if (frozen>0) {
                    for(size_t i=0;i<sleeping.size();++i)
                        if (sleeping[i]) ff_monode::getlb()->thaw(i, false); 
                }
                printf("EMITTER2 BROADCASTING EOS\n");
                broadcast_task(EOS);
            }
        } 
    }
private:
    bool eos_received = 0;
    unsigned last, nready, frozen=0, onthefly=0;
    std::vector<bool>  ready;         // which workers are ready
    std::vector<bool>  sleeping;      // which workers are sleeping
    std::vector<long*> data;          // local storage
};

struct Worker: ff_monode_t<long> {
    int svc_init() {
        printf("Worker id=%ld starting\n", get_my_id());
        return 0;
    }

    long* svc(long* task) {
        //printf("Worker id=%ld got %ld\n", get_my_id(), (long)task);
        ff_send_out_to(task, 1);  // to the next stage 
        ff_send_out_to(task, 0);  // send the "ready msg" to the emitter 
        return GO_ON;
    }

    void eosnotify(ssize_t) {
        printf("Worker2 id=%ld received EOS\n", get_my_id());
    }
    
    void svc_end() {
        //printf("Worker2 id=%ld going to sleep\n", get_my_id());
    }

};

// multi-input stage
struct Collector: ff_minode_t<long> {
    long* svc(long* task) {
        //printf("Collector received task = %ld, sending it back to the Emitter\n", (long)(task));
        return task;
    }
    void eosnotify(ssize_t) {
        printf("Collector received EOS\n");
    }
    
};


struct Manager: ff_node_t<Command_t> {
    Manager(): 
        channel(100, true, MANAGERID) {

        // The following is needed if the program runs in BLOCKING_MODE
        // In this case, channel will not be automatically initialized
        // so we have to do it manually. Moreover, even if we
        // want to run in BLOCKING_MODE, channel cannot be
        // configured in blocking mode for the output, that is, ff_send_out
        // has to be non-blocking!!!!
        channel.init_blocking_stuff();
        channel.reset_blocking_out();        
    }
    
    Command_t* svc(Command_t *) {
        
        struct timespec req = {0, static_cast<long>(5*1000L)};
        nanosleep(&req, NULL);

        Command_t *cmd1 = new Command_t(0, REMOVE);
        channel.ff_send_out(cmd1);

        Command_t *cmd2 = new Command_t(1, REMOVE);
        channel.ff_send_out(cmd2);

        {
            struct timespec req = {0, static_cast<long>(5*1000L)};
            nanosleep(&req, NULL);
        }

        Command_t *cmd3 = new Command_t(1, ADD);
        channel.ff_send_out(cmd3);

        Command_t *cmd4 = new Command_t(0, ADD);
        channel.ff_send_out(cmd4);

        {
            struct timespec req = {0, static_cast<long>(5*1000L)};
            nanosleep(&req, NULL);
        }

        channel.ff_send_out(EOS);

        return GO_OUT;
    }

    void svc_end() {
        printf("Manager ending\n");
    }


    int run(bool=false) { return ff_node_t<Command_t>::run(); }
    int wait()          { return ff_node_t<Command_t>::wait(); }


    ff_buffernode * getChannel() { return &channel;}

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

    std::vector<ff_node* > W;
    for(size_t i=0;i<nworkers;++i)  
        W.push_back(new Worker);
    ff_farm farm(W);
    farm.cleanup_workers();
    
    // registering the manager channel as one extra input channel for the load balancer
    farm.getlb()->addManagerChannel(manager.getChannel());
    
    Emitter E;

    farm.remove_collector();
    farm.add_emitter(&E); 
    farm.wrap_around();
    // here the order of instruction is important. The collector must be
    // added after the wrap_around otherwise the feedback channel will be
    // between the Collector and the Emitter
    Collector C;
    farm.add_collector(&C);
    farm.wrap_around();

    ff_Pipe<> pipe(seq, farm);

    if (pipe.run_then_freeze()<0) {
        error("running pipe\n");
        return -1;
    }            
    manager.run();
    pipe.wait_freezing();
    pipe.wait();
    manager.wait();

    return 0;
}
