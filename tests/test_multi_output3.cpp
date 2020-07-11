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
 *  A strange skeleton implemented by using building-blocks
 *   
 *    pipe(E, feedback(A2A))   
 *
 *                               ---------------------
 *                               v                    |
 *                             -----                  |
 *                       ---->|     |---------        |
 *                      |     |  W  |        |        |
 *                      |      -----         v        |
 *             -----    |        .          -----     |
 *            |  E  |-->| -->    .     --> |  C  | ---|
 *             -----    |        .          -----     |
 *                      |      -----         ^        | 
 *                       ---->|  W  |        |        | 
 *                            |     |--------         |
 *                             -----                  |
 *                              ^                     |
 *                              ----------------------
 *
 * NOTE:
 *  - In front of workers there are multiInputHelper(s)
 *  - The collector uses a multiOutputHeler
 */      
/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

static const long ITER = 200;

struct Emitter: ff_monode_t<long> {
    long* svc(long *) {
        for(long i=1;i<=ITER;++i)
            ff_send_out((long*)i);
        return EOS;
    }
};

using mypair = std::pair<long,long>;
struct Collector: ff_minode_t<long, mypair> {

    mypair* svc(long *in) {
        long id = get_channel_id();
        mypair *p = new mypair;
        p->first  = (long)in;
        p->second = id;        
        return p;
    }
};
struct multiOutputHelper: ff_monode_t<mypair, long> {
    long* svc(mypair *in) {
        ff_send_out_to((long*)(in->first), in->second);
        delete in;
        return GO_ON;
    }
};

struct multiInputHelper: ff_minode_t<long, mypair> {
    mypair* svc(long *in) {
        mypair *p = new mypair;
        p->first  = (long)in;
        p->second = (fromInput()?-1:0);
        return p;
    }
    void eosnotify(ssize_t) {
        // NOTE: we have to send EOS explicitly to the next stage
        // because for multi-input node the EOS is propagated only
        // it has been received from all input channels
        ff_send_out(EOS);
    }
};
// this node must be multi-output even if  it is connected only to 
// one node
struct Worker: ff_monode_t<mypair, long> { 
    long* svc(mypair *in) {
        if (in->second == -1) {
            ntasks++;

            // introducing a minimal delay
            if (get_my_id()== 0 || get_my_id() == 1)
                usleep(50000);
            
            ff_send_out((long*)(in->first));
            delete in;
        } else {
            printf("Worker received %ld back \n",
                   (long)in->first);
            --ntasks;
            assert(ntasks>=0);
            delete in;
            if (ntasks == 0 && eosarrived) return EOS;
        }            
        return GO_ON;       
    }
    void eosnotify(ssize_t) {
        eosarrived=true;
        if (eosarrived && ntasks==0)
            ff_send_out(EOS);
    }
    long ntasks=0;
    bool eosarrived=false;
};


int main() {
    Emitter E;

    // creating 3 workers 
    const Worker W;
    const multiInputHelper h1;

    auto comp1 = combine_nodes(h1,W);
    auto comp2 = combine_nodes(h1,W);
    auto comp3 = combine_nodes(h1,W);

    std::vector<ff_node*> W1;
    W1.push_back(&comp1);
    W1.push_back(&comp2);
    W1.push_back(&comp3);
    
    Collector C;
    multiOutputHelper h2;
    auto comp = combine_nodes(C, h2);
    std::vector<ff_node*> W2;
    W2.push_back(&comp);
    
    ff_a2a a2a;
    if (a2a.add_firstset(W1)<0) {
        error("adding first set of workers failed\n");
        return -1;
    }
    if (a2a.add_secondset(W2)<0) {
        error("adding second set of workers failed\n");
        return -1;
    }
    if (a2a.wrap_around()<0) {      
        error("wrap_around failed\n");
        return -1;
    }
    
    ff_Pipe<> pipe(E, a2a);
    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }
    printf("TEST DONE\n");
    return 0;

}

