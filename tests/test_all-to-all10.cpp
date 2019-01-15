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
/*  feedback( A2A )
 *
 *     |<- comp(Filter1, Filter2) ->|   |<- comp(Filter3,Filter4) ->|  
 *   ________________________________________________________________
 *  |                                                                ^
 *  |                                                                |
 *  |    |-> Filter1 ---> Filter2 --->|                              |
 *  v    |                            | ---> Filter3 ---> Filter4 -->|
 *  |--> |-> Filter1 ---> Filter2 --->|                              
 *  ^    |                            | ---> Filter3 ---> Filter4 -->|
 *  |    |-> Filter1 ---> Filter2 --->|                              |
 *  |                                                                |
 *  |_________________________________________feedback_______________v 
 *
 *
 *
 */
/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

// NOTE: the Filter1 can be defined either as ff_node or as ff_minode
//       If it is a multi-input node it is possible to discern who is the sender
//       (by using get_channel_id()).
struct Filter1: ff_minode_t<long> {   // multi-input
	long *svc(long *in) {
        if (in == nullptr) {
            for(size_t i=1; i<=ntasks; ++i) {
                ff_send_out((long*)i);
            }
            ff_send_out(EOS);
            return GO_ON;
        }
        std::cout << "Filter1: got back result from " << get_channel_id() << "\n";
        return GO_ON;
    }
    size_t ntasks=10;
    bool check;
};

struct Filter2: ff_monode_t<long> {   // multi-output
	long *svc(long *in) {
        std::cout << "Filter2: (" << get_my_id() << "): " << (long)in << "\n";
        return in;
    }
};

struct Filter3: ff_minode_t<long> {   // multi-input
	long *svc(long *in) {
        std::cout << "Filter3: (" << get_my_id() << "): sending back " << (long)in << "\n";
        return in;
    }
};
struct Filter4: ff_monode_t<long> {   // multi-output
	long *svc(long *in) {
        std::cout << "Filter3: (" << get_my_id() << "): sending back " << (long)in << "\n";
        return in;
    }
};

int main() {

    Filter1 f11;
    Filter2 f21;
    auto comb1 = combine_nodes(f11, f21);

    Filter1 f12;
    Filter2 f22;
    auto comb2 = combine_nodes(f12, f22);
    
    Filter1 f13;
    Filter2 f23;
    auto comb3 = combine_nodes(f13, f23);

    std::vector<ff_node*> W1;  
    W1.push_back(&comb1);
    W1.push_back(&comb2);
    W1.push_back(&comb3);

    Filter3 f31, f32;
    Filter4 f41, f42;

    auto comb4 = combine_nodes(f31, f41);
    auto comb5 = combine_nodes(f32, f42);
    
    std::vector<ff_node*> W2;          
    W2.push_back(&comb4);
    W2.push_back(&comb5);
    
    ff_a2a a2a;
    a2a.add_firstset(W1);
    a2a.add_secondset(W2);
    
    a2a.wrap_around();

    if (a2a.run_and_wait_end()<0) {
        error("running A2A\n");
        return -1;
    }
    
    return 0;
}
