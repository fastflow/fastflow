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
/*  feedback( pipe(A2A,farm) )
 *
 *    _______________________________________________________________________
 *   |                                                                       |
 *   |      | --> Filter1 -->|                                |--> Worker -->|
 *   |      |                | --> Filter2 -->|               |              |
 *   |----> | --> Filter1 -->|                | --> Emitter-->|--> Worker -->|
 *   |      |                | --> Filter2 -->|               |              |
 *   |      | --> Filter1 -->|                                |--> Worker -->|
 *   |_______________________________________________________________________|
 *
 * NOTE: if the cardinality of farm's workers is different from the cardinality of the 
 *       first set of nodes in the A2A, then farm's Workers must be multi-output nodes. 
 *       Otherwise, if the cardinality is the same, farm's Workers can be standard 
 *       ff_nodes. 
 *       (see also comments in test_all-to-all8.cpp)
 *
 */
/* Author: Massimo Torquati
 *
 */
               
#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

using mypair = std::pair<long,long>;

struct Filter1: ff_monode_t<mypair> {
    Filter1(size_t nfilter2, bool check=false):nfilter2(nfilter2),check(check) {}

    int svc_init() {
        ntasks = 10*nfilter2;
        return 0;
    }
    
	mypair *svc(mypair *in) {
        if (in == nullptr || (in->first == -1 && in->second == -1) ) {
            for(size_t i=0; i<ntasks; ++i) {
                mypair *out = new mypair;
                out->first = get_my_id();
                out->second = i;
                ff_send_out_to(out, i%nfilter2);
            }
            return GO_ON;
        }
        if (check) {
            if (in->first != get_my_id()) {
                error("WRONG INPUT DATA: %ld vs %ld\n", in->first, get_my_id());
                abort();
            }
        }
        delete in;
        if (--ntasks == 0) {
            return EOS;
        }
        return GO_ON;
    }
    size_t nfilter2;
    size_t ntasks;
    bool check;
};

struct Filter2: ff_node_t<mypair> {
	mypair *svc(mypair *in) {
        //  std::cout << get_my_id() << " : " << in->second << " from: " << in->first << "\n";
        return in;
    }
};

struct Emitter: ff_monode_t<mypair> {
    Emitter(int nworkers):nworkers(nworkers) {}
	mypair *svc(mypair *in) {
        ff_send_out_to(in, in->first % nworkers);
        return GO_ON;
    }
    int nworkers;
};

int main() {
    int nfilter1 = 3;
    int nfilter2 = 2;
    int nworkers = 2;

    { // first test, workers are multi-output

        struct Worker: ff_monode_t<mypair> {
            mypair *svc(mypair *in) {
                printf("Worker %ld sending data (%ld) to %ld\n", get_my_id(), in->second, in->first);
                ff_send_out_to(in, in->first);
                return GO_ON;
            }
        };
        struct MultiInputHelper: ff_minode_t<mypair> {
            mypair *svc(mypair *in) {
                if (in == nullptr) {
                    // generates a special mypair to let the
                    // Filter1 to start
                    mypair *init=new mypair;
                    init->first=-1;
                    init->second=-1;
                    return init;
                }
                printf("HELPER got data back\n");
                return in;
            }
        };

        const MultiInputHelper helper;
        const Filter1          F1(nfilter2,true);
        std::vector<ff_node*> W1;  
        for(int i=0;i<nfilter1;++i)
            W1.push_back(new ff_comb(helper, F1));
        std::vector<ff_node*> W2;          
        for(int i=0;i<nfilter2;++i)
            W2.push_back(new Filter2);
        
        ff_a2a a2a;
        a2a.add_firstset(W1);
        a2a.add_secondset(W2);

        Emitter E(nworkers);
        std::vector<ff_node*> W;
        for(int i=0;i<nworkers;++i)
            W.push_back(new Worker);
        ff_farm farm(W, &E);
        farm.remove_collector();

        ff_Pipe<> pipe(a2a, farm);
        if (pipe.wrap_around()<0) {
            error("wrap_around\n");
            return -1;
        }
        
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }

        for(int i=0;i<nfilter1;++i) delete W1[i];
        for(int i=0;i<nfilter2;++i) delete W2[i];
        for(int i=0;i<nworkers;++i) delete W[i];
    }
    printf("TEST1 DONE\n");
    sleep(1);
    { // second test, workers are standard node

        // the cardinality must be the same in this case
        nworkers = nfilter1;

        
        struct Worker: ff_node_t<mypair> {
            mypair *svc(mypair *in) {
                printf("Worker %ld sending data (%ld) back\n", get_my_id(), in->second);
                return in;
            }
        };
                
        std::vector<ff_node*> W1;  
        for(int i=0;i<nfilter1;++i)
            W1.push_back(new Filter1(nfilter2));
        std::vector<ff_node*> W2;          
        for(int i=0;i<nfilter2;++i)
            W2.push_back(new Filter2);
        
        ff_a2a a2a;
        a2a.add_firstset(W1);
        a2a.add_secondset(W2);

        Emitter E(nworkers);
        std::vector<ff_node*> W;
        for(int i=0;i<nworkers;++i)
            W.push_back(new Worker);
        ff_farm farm(W, &E);
        farm.remove_collector();

        ff_Pipe<> pipe(a2a, farm);
        if (pipe.wrap_around()<0) {
            error("wrap_around\n");
            return -1;
        }
        
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }

        for(int i=0;i<nfilter1;++i) delete W1[i];
        for(int i=0;i<nfilter2;++i) delete W2[i];
        for(int i=0;i<nworkers;++i) delete W[i];
    }
    printf("TEST2 DONE\n");
    return 0;
}
