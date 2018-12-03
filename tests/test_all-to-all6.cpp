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
/*  pipe(farm, A2A)
 *
 *
 *  |<----------------- farm ----------------->|   |<------ all-to-all ------>|   
 *  |            with a multi-output           |
 *  |                  collector               |
 *
 *
 *              |-> Worker ->|                 | --> Filter1 -->|
 *              |            |                 |                | --> Filter2
 *   Emitter -> |-> Worker ->|-->mo-Collector->| --> Filter1 -->|
 *              |            |                 |                | --> Filter2
 *              |-> Worker ->|                 | --> Filter1 -->|
 *
 */
/* Author: Massimo Torquati
 *
 */
               
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

static long ntasks=100;

struct Emitter: ff_monode_t<long> { 
    long *svc(long*) {
        for(long i=1;i<=ntasks;++i) {
            ff_send_out_to((long*)i, i % 3);
        }
        return EOS;
    }
};
struct Worker: ff_node_t<long> {
    long *svc(long *in) {
        return in;
    }
};
struct moCollector: ff_monode_t<long> {
    long *svc(long *in) {
        return in;
    }
};

struct Filter1: ff_monode_t<long> {
	long *svc(long *in) {
        return in;
    }
};

struct Filter2: ff_node_t<long> {
	long *svc(long *in) {
        std::cout << get_my_id() << ": " << (long)in << "\n";
        return GO_ON;
    }
};

int main() {

    Emitter E;
    moCollector C;
    
    std::vector<std::unique_ptr<ff_node>> W;
    W.push_back(make_unique<Worker>());
    W.push_back(make_unique<Worker>());
    W.push_back(make_unique<Worker>());
    
    ff_Farm<> farm(std::move(W), E, C);

    std::vector<ff_node*> W1;  
    Filter1 f11, f12, f13;
    W1.push_back(&f11);
    W1.push_back(&f12);
    W1.push_back(&f13);
    std::vector<ff_node*> W2;          
    Filter2 f21,f22;
    W2.push_back(&f21);
    W2.push_back(&f22);
    
    ff_a2a a2a;
    a2a.add_firstset(W1);
    a2a.add_secondset(W2);
    
    ff_Pipe<> pipe(farm, a2a);
        
    if (pipe.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    
    return 0;
}
