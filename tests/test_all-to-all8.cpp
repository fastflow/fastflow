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
/*  feedback(A2A)
 *
 *    ________________________________________
 *   |                                        |
 *   |      | --> Filter1 -->|                |
 *   |      |                | --> Filter2 -->|
 *   |----> | --> Filter1 -->|                |
 *   |      |                | --> Filter2 -->|
 *   |      | --> Filter1 -->|                |
 *   |________________________________________|
 *
 * 2 tests:
 *   test1: different cardinality of the first and second set
 *   test2: n. of filter1 nodes = n. of filter2 nodes
 * 
 *  NOTE: In the first test, since both Filter1 and Filter2 are multi-output nodes (and because of feedback channel), we need a multi-input helper node in front of both of them.
 *
 *
 */
/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

using mypair = std::pair<long,long>;

/* --------------------------------- */
// these are the helper nodes. They are needed in the first test to transform a
// multi-output node in a node that is both multi-input and multi-output
// (i.e. a composition of two nodes, the first one multi-input and the second
// one multi-output).
struct MultiInputHelper1: ff_minode_t<mypair> {
	mypair *svc(mypair *in) {
        if (in == nullptr) return &init;
        return in;
    }
    mypair init{-1,-1};
};
struct MultiInputHelper2: ff_minode_t<mypair> {
	mypair *svc(mypair *in) {
        return in;
    }
};
/* --------------------------------- */


struct Filter1: ff_monode_t<mypair> {
    Filter1(size_t nfilter2, bool check=false):nfilter2(nfilter2),check(check) {}

    int svc_init() {
        ntasks= 10*nfilter2;
        return 0;
    }
    
	mypair *svc(mypair *in) {
        if (in->first==-1 && in->second == -1) {
            for(size_t i=0; i<ntasks; ++i) {
                mypair *out = new mypair;
                out->first = get_my_id();
                out->second = i;
                ff_send_out_to(out, i%nfilter2);
            }
            return GO_ON;
        }
        if (check) {
            if (in->first != get_my_id()) abort();
        }

        printf("Filter1 (%ld) got back result\n", get_my_id());
        
        delete in;
        if (--ntasks == 0) return EOS;
        return GO_ON;
    }
    void svc_end() { assert(ntasks==0); }
    
    size_t nfilter2;
    size_t ntasks=0;
    bool check;
};
struct Filter2: ff_monode_t<mypair> {
	mypair *svc(mypair *in) {
        std::cout << "Filter2 (" << get_my_id() << ") : " << in->second << " from: " << in->first << "\n";
        ff_send_out_to(in, in->first);
        return GO_ON;
    }
};


struct Filter12: ff_monode_t<mypair> {
    Filter12(size_t nfilter2, bool check=false):nfilter2(nfilter2),check(check) {}

    int svc_init() {
        ntasks= 10*nfilter2;
        return 0;
    }
    
	mypair *svc(mypair *in) {
        if (in == nullptr || (in->first==-1 && in->second == -1)) {
            for(size_t i=0; i<ntasks; ++i) {
                mypair *out = new mypair;
                out->first = get_my_id();
                out->second = i;
                ff_send_out_to(out, get_my_id());
            }
            delete in;
            return GO_ON;
        }
        if (check) {
            if (in->first != get_my_id()) abort();
        }

        printf("Filter12 (%ld) got back result\n", get_my_id());
        
        delete in;
        if (--ntasks == 0) return EOS;
        return GO_ON;
    }
    void svc_end() { assert(ntasks==0); }
    
    size_t nfilter2;
    size_t ntasks=0;
    bool check;
};
// this is a standard node
struct Filter22: ff_node_t<mypair> {
	mypair *svc(mypair *in) {
        std::cout << "Filter22 (" << get_my_id() << ") : " << in->second << " from: " << in->first << "\n";
        return in;
    }
};


int main() {
    int nfilter1 = 3; 
    int nfilter2 = 2;

    { // first test

        // NOTE: defining them as const is very important here because
        //       we have to create a different composition for each node
        const MultiInputHelper1 helper1;
        const MultiInputHelper2 helper2;
        const Filter1          F1(nfilter2,true);
        const Filter2          F2;
        
        std::vector<ff_node*> W1;
        for(int i=0;i<nfilter1;++i) {
            // dynamically creating a composition of two nodes 
            W1.push_back(new ff_comb(helper1, F1));
        }

        std::vector<ff_node*> W2;          
        for(int i=0;i<nfilter2;++i) {
            // dynamically creating a composition of two nodes 
            W2.push_back(new ff_comb(helper2, F2));
        }
        
        ff_a2a a2a;
        a2a.add_firstset(W1,0,true);
        a2a.add_secondset(W2, true);
        if (a2a.wrap_around()<0) {
            error("wrap_around\n");
            return -1;
        }
        
        if (a2a.run_and_wait_end()<0) {
            error("running A2A\n");
            return -1;
        }
    }
    printf("TEST1 DONE\n");
    sleep(1);
    { // second test

        // we force the same cardinality for the 2 sets
        nfilter2 = nfilter1;
        
        std::vector<ff_node*> W1;  
        for(int i=0;i<nfilter1;++i)
            W1.push_back(new Filter12(nfilter2, true));
        std::vector<ff_node*> W2;          
        for(int i=0;i<nfilter2;++i)
            W2.push_back(new Filter22);
        
        ff_a2a a2a;
        a2a.add_firstset(W1);
        a2a.add_secondset(W2);
        if (a2a.wrap_around()<0) {
            error("wrap_around\n");
            return -1;
        }
        
        if (a2a.run_and_wait_end()<0) {
            error("running A2A\n");
            return -1;
        }

        for(int i=0;i<nfilter1;++i) delete W1[i];

        for(int i=0;i<nfilter2;++i) delete W2[i];
    }
    printf("TEST2 DONE\n");
    return 0;
}
