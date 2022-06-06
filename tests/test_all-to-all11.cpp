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
 *
 *
 */

/*  pipe( First, feedback(farm(A2A), Last )
 *
 *   - the farm has one worker whose type is A2A, Worker is multi-output
 *   - Last is multi-input 
 *
 *                         |<------------- A2A ------------>|            
 *                   _________________________________________
 *                  |                                         ^
 *                  |                      | -> Worker -->|   |
 *                  v       |-> Emitter2 ->|              |   |
 *     First ---> Emitter ->|              | -> Worker -->| ->| ---> Last
 *                  ^       |-> Emitter2 ->|              |   |
 *                  |                      | -> Worker -->|   |
 *                  |_________________________________________v
 *
 *  Another way to implement this topology is to split the Emitter in two nodes:
 *  Gatherer and Router, where Gatherer is multi-input and Router is multi-output,
 *  then it is possible to use the comp building block to merge them in one node:
 *
 *     pipe( First, feedback(combine_nodes(Gather,Router), A2A), Last)
 * 
 *  A variant (implemented below) for routing back the data directly to the Emitter2 is to use 
 *  a Gatherer node before the Emitter2 nodes:
 *
 *     pipe( First, Emitter, feedback(A2A), Last)
 *
 *   NOTE: in this case program termination is much more complex to manage because of we could
 *         have more than one workers in the fist set (i.e. more than one Gatherer-Emitter2
 *         composition).
 * 
 *
 *                              |<-------------- feedback(A2A) ---------------->|            
 *                                 |<-------- comp ------->|
 *                           ___________________________________________________
 *                          |                                                   ^
 *                          |                                | -> Worker -->|   |
 *                          |--> |-> Gatherer --> Emitter2 ->|              |   |
 *     First ---> Emitter ------>|                           | -> Worker -->| ->| ---> Last
 *                          |--> |-> Gatherer --> Emitter2 ->|              |   |
 *                          |                                | -> Worker -->|   |
 *                          |___________________________________________________v
 *
 * NOTE: Since Worker is multi-output we need to prepend to such node a multi-input
 *       helper node because a single node cannot be at the same time multi-input
 *       and multi-output (unless that node is a composition of two nodes).
 *       (see test_all-to-all8/9.cpp) 
 *
 */

/* Author: Massimo Torquati
 *
 */

#include <iostream>

#include <ff/ff.hpp>
using namespace ff;

// used to manage termination!!!!!
std::atomic<long> ntasks;

struct First: ff_node_t<long> {
    long* svc(long*) {
        long n = ntasks.load();
        for(long i=0;i<n;++i)  
            ff_send_out(new long(i));
        return EOS;
    }
};

struct Emitter: ff_monode_t<long> {
    long* svc(long* in) {
        return in;
    }  
};

struct Gatherer: ff_minode_t<long> {
    long* svc(long* in) {
        if (!fromInput()) {
            long n=ntasks.fetch_sub(1);
            if (n == 1){
                return EOS;
            }
            return GO_ON;
        }
        return in;
    }
};

struct Emitter2: ff_monode_t<long> {
    long* svc(long* in) {
        return in;
    }

    void eosnotify(ssize_t) {
        // WARNING: this is needed because Emitter2 is part of a composition that
        //          is a multi-input node and so the EOS is not propagated
        //          until all EOSs from all input channels are received 
        broadcast_task(EOS);
    }
};

struct MultiInputHelper: ff_minode_t<long> {
	long *svc(long *in) {
        return in;
    }
    void eosnotify(ssize_t) {
        // propagate EOS
        ff_send_out(EOS);
    }
};

struct Worker: ff_monode_t<long> {
    long* svc(long* in) {
        std::cout << "Worker" << get_my_id() << " got " << *in << "\n";

        // Here we know that we have only one output channel
        // and we have 'get_num_feedbackchannels()' feedback channels.
        // Since the numbering of the channels is: first the feedback ones
        // and then the output ones .....
        ff_send_out_to(in, get_num_feedbackchannels() ); // to Last

        //std::cout << "Worker" << get_my_id() << " sending ack back\n";
        
        ff_send_out((long*)0x1); // sends it back

        return GO_ON;
    }

    void eosnotify(ssize_t) {
        //broadcast_task(EOS); // sending EOS back
        ff_send_out_to(EOS, get_num_feedbackchannels());
    }
};

struct Last: ff_minode_t<long> {
    Last(size_t nt):nt(nt) {}
    long* svc(long* in) {
        std::cout << "Last: received " << *in << " from " << get_channel_id() << "\n";
        --nt;
        delete in;
        return GO_ON;
    }
    void svc_end() {
        if (nt != 0) {
            std::cerr << "Test FAILED\n";
            exit(-1);
        }
        std::cout << "Test OK\n";
    }
    size_t nt;
};

int main() {
    int nworkers = 3;
    ntasks.store(100);

    // ---- first stage
    First first;

    // ---- second stage
    Emitter E;
    
    // ----- building all-to-all
    std::vector<ff_node*> firstSet;  
    Gatherer gt1;
    Emitter2 e1;
    auto comb1 = combine_nodes(gt1, e1);
    firstSet.push_back(&comb1);
    Gatherer gt2;
    Emitter2 e2;
    auto comb2 = combine_nodes(gt2, e2);
    firstSet.push_back(&comb2);

    const MultiInputHelper helper;
    const Worker           worker;
    std::vector<ff_node*> secondSet;          
    for(int i=0;i<nworkers;++i)
        secondSet.push_back(new ff_comb(helper,worker));
    
    ff_a2a a2a;
    a2a.add_firstset(firstSet);
    a2a.add_secondset(secondSet, true);
    a2a.wrap_around();

    // ---- last stage
    Last last(ntasks.load());

    // ---- building the topology
    ff_Pipe<> pipe(first, E, a2a, last);

    //******** 
    
    if (pipe.run_and_wait_end() <0) {
        error("running pipe\n");
        return -1;
    }
    return 0;
}
