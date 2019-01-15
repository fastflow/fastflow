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
/*  pipe(A2A_1, A2A_2)
 *
 *
 *  |<----- all-to-all ----->|   |<------ all-to-all ------->|   
 * 
 *
 *   Generator-->|---> Filter1 ----> Filter2 -->
 *               |                              | --> Filter3
 *               |                              |
 *   Generator-->|---> Filter1 ----> Filter2 -->|
 *               |                              | --> Filter3
 *               |                              |
 *   Generator-->|---> Filter1 ----> Filter2 -->
 *
 *
 * NOTE: if Filter1 is a standard node (only one output channel), then
 *       the number of Filter1 nodes must be equal to the number of 
 *       Filter2 nodes.
 *
 *       To have different cardinality of nodes between the two pipe stages,
 *       Filter1 should be multi-output and Filter2 must be a multi-input
 *       node, i.e. the first set of nodes of A2A_2 must be replaced by 
 *       a composition of a multi-input helper node and the Filter2 node
 *       (see test_all-to-all7.cpp).  
 */
/* Author: Massimo Torquati
 *
 */
               
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

struct Generator: ff_monode_t<long> { 
    long *svc(long*) {
        int myid= get_my_id();
        long start = myid*10 + 1;
        long stop  = start + 10 + 1;
        for(long i=start;i<stop;++i) {
            ff_send_out_to((long*)i, i % 3);
        }
        return EOS;
    }
};
struct Filter1: ff_node_t<long> {  
    long *svc(long *in) {
        printf("Filter1 received %ld\n", (long)in);
        return in;
    }
};

struct Filter2: ff_monode_t<long> {
    long *svc(long *in) {
        printf("Filter2 received %ld\n", (long)in);
        return in;
    }
};

struct Filter3: ff_node_t<long> {
	long *svc(long *in) {
        std::cout << get_my_id() << ": " << (long)in << "\n";
        return GO_ON;
    }
};

int main() {

    std::vector<ff_node*> W1;  
    Generator g1, g2, g3;
    W1.push_back(&g1);
    W1.push_back(&g2);
    W1.push_back(&g3);
    std::vector<ff_node*> W2;
    Filter1 f11, f12, f13;
    W2.push_back(&f11);
    W2.push_back(&f12);
    W2.push_back(&f13);

    ff_a2a a2a_1;
    a2a_1.add_firstset(W1);
    a2a_1.add_secondset(W2);

    W1.clear();
    W2.clear();
    Filter2 f21, f22, f23;
    W1.push_back(&f21);
    W1.push_back(&f22);
    W1.push_back(&f23);
    Filter3 f31,f32;
    W2.push_back(&f31);
    W2.push_back(&f32);
    
    ff_a2a a2a_2;
    a2a_2.add_firstset(W1);
    a2a_2.add_secondset(W2);

    
    ff_Pipe<> pipe(a2a_1, a2a_2);
        
    if (pipe.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    
    return 0;
}
