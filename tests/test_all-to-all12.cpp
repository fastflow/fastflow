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

/*  pipe( First, feedback(A2A), Last )
 *
 *                     |<------------- A2A ------------>|            
 *                   _______________________________________
 *                  |                                       ^
 *                  |                   | -> Worker2 -->|   |
 *                  v    |-> Worker1  ->|               |   |
 *     First ---->  |--->|              | -> Worker2 -->| ->| -----> Last
 *                  ^    |-> Worker1  ->|               |   |
 *                  |                   | -> Worker2 -->|   |
 *                  |_______________________________________v
 *
 *  |<- multi- ->|                                           |<- multi- ->|
 *     output                                                    input
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

int NITER = 500;

struct First: ff_monode_t<long> {
    First(int nworkers):nworkers(nworkers) {}
    long* svc(long*) {
        for(int i=0;i<nworkers; ++i)
            ff_send_out_to(new long(0), i);
        return EOS;
    }
    int nworkers;
};


struct MultiInputHelper: ff_minode_t<long> {
	long *svc(long *in) { return in; }
};

struct Worker1: ff_monode_t<long> {
    long *svc(long *in) {
        long &i = *in;
        if (i == NITER) {
            return EOS;
        }
        ++i;
        if (get_my_id()==1) usleep(4000);
        ff_send_out_to(in, get_my_id());
        return GO_ON;
    }
};

struct MultiInputHelper2: ff_minode_t<long> {
	long *svc(long *in) { return in; }
    void eosnotify(ssize_t) {
        if (++neos == get_num_inchannels())
            ff_send_out(EOS);        
    }
    size_t neos=0;
};

struct Worker2: ff_monode_t<long> {    
    long* svc(long* in) {
        last = in;
        if (get_my_id() == 0) usleep(5000);
        ff_send_out_to(in, get_my_id()); // sends it back
        return GO_ON;
    }

    void eosnotify(ssize_t) {
        ff_send_out_to(last, get_num_feedbackchannels() ); 
        ff_send_out_to(EOS, get_num_feedbackchannels());
    }

    long *last = nullptr;;
};

struct Last: ff_minode_t<long> {
    long* svc(long* in) {
        if (*in != NITER) {
            std::cerr << "Last: received the WRONG value\n";
            abort();
        }
        std::cout << "Last: received the correct value (" << *in << ") from " << get_channel_id() << "\n";
        delete in;
        return GO_ON;
    }
};

int main() {
    // ---- first stage
    First first(3);

    // ----- building all-to-all
    std::vector<ff_node*> firstSet;  

    const MultiInputHelper helper;
    const MultiInputHelper2 helper2;
    const Worker1 w1;
    auto comb1 = combine_nodes(helper, w1);
    auto comb2 = combine_nodes(helper, w1);
    auto comb3 = combine_nodes(helper, w1);
    firstSet.push_back(&comb1);
    firstSet.push_back(&comb2);
    firstSet.push_back(&comb3);

    const Worker2         w2;
    std::vector<ff_node*> secondSet;              
    secondSet.push_back(new ff_comb(helper2,w2));        
    secondSet.push_back(new ff_comb(helper2,w2));        
    secondSet.push_back(new ff_comb(helper2,w2));
    
    ff_a2a a2a;
    a2a.add_firstset(firstSet);
    a2a.add_secondset(secondSet, true);
    a2a.wrap_around();

    // ---- last stage
    Last last;

    // ---- building the topology
    ff_Pipe<> pipe(first, a2a, last);

    if (pipe.run_and_wait_end() <0) {
        error("running pipe\n");
        return -1;
    }
    return 0;
}
