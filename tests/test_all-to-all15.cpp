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

/* Testing an all-to-all whose R-Workers are all-to-all 
 * (i.e. a pipeline whose first stage is an A2A). 
 *
 *                                              
 *  | --> Source1 -->| 
 *  |                |    |-->mi-> Worker1->mo -->|
 *  |                | -->|                       |--> Sink
 *  |                |    |-->mi-> Worker1->mo -->|
 *  | --> Source2 -->|
 * 
 *                        |<--- pipeline ---------->|
 *                        |<------- all-to-all ----------->|
 *  |<---------------- all-to-all ------------------------>|
 *  |<---------------- pipeline -------------------------->|
 *
 *
 */

/*
 * Author: Massimo Torquati
 */ 

#include <ff/ff.hpp>

using namespace ff;

const long ntasks=100000;
struct Source: ff_monode_t<long> {
    long* svc(long*) {
        for(long i=1;i<=ntasks;++i) {
            ff_send_out((long*)i);
        }
        return EOS;
    }
};

struct PipeA2A: ff_pipeline {
    struct Sink: ff_minode_t<long> {        
        long* svc(long*) {
            ++cnt;
            return GO_ON;
        }
        long cnt=0;
    };
    struct miHelper:ff_minode_t<long> {
        long* svc(long* in) {
            return in;
        }
        void eosnotify(ssize_t) {
            printf("miHelper %ld received EOS\n", get_my_id());
        }
    };   
    struct Worker:ff_node_t<long> {
        long* svc(long* in) {
            printf("Worker%ld received task %ld\n", get_my_id(), (long)in);
            return in;
        }
        void eosnotify(ssize_t) {
            printf("Worker %ld received EOS\n", get_my_id());
        }
    };
    struct moHelper:ff_monode_t<long> {
        long* svc(long* in) {
            return in;
        }    
        void eosnotify(ssize_t) {
            printf("moHelper %ld received EOS\n", get_my_id());
        }
    };

    PipeA2A(int /*nsources*/) {
        const long nworkers=2;

        ff_a2a* a2a = new ff_a2a;
        std::vector<ff_node*> W1;
        for(long int i=0;i<nworkers;++i) {
            ff_comb *t  = new ff_comb(new miHelper, new Worker, true, true);
            ff_comb *w1 = new ff_comb(t, new moHelper, true, true);
            W1.push_back(w1);
        }
        a2a->add_firstset(W1, 0, true);
        
        std::vector<ff_node*> W2;
        sink = new Sink;
        assert(sink);
        W2.push_back(sink);
        a2a->add_secondset(W2,true);
        add_stage(a2a, true);
    }
    Sink*sink=nullptr;
};


int main() {
    const long nsources= 3;
    ff_a2a a2a;
    std::vector<ff_node*> W1;
    for(long i=0;i<nsources;++i)
        W1.push_back(new Source);
    a2a.add_firstset(W1,0,true);
    std::vector<ff_node*> W2;
    PipeA2A a2aI(nsources);
    W2.push_back(&a2aI);
    a2a.add_secondset(W2,false);
    
    ff_Pipe<> pipe(a2a);
    
    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }
    if (a2aI.sink->cnt != nsources*ntasks) {
        printf("TEST FAILED\n");
        return -1;
    } printf("TEST OK\n");
    return 0;
}
