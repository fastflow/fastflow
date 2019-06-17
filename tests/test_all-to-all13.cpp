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

/*  pipe( feedback(farm), A2A, Last )
 *
 *
 *         -----------                        |-> Worker3  ---- 
 *        |           |                       |                |
 *        |        Worker1  --->  Worker2 --> |                |
 *        v                                   |                v
 *     Emitter                                |-> Worker3  ------->  Last
 *        ^                                   |                ^
 *        |        Worker1  --->  Worker2 --> |                |
 *        |           |                       |                |
 *         -----------                        |-> Worker3 -----
 *
 *      |------ farm -----|     |--------------- A2A --------------|
 *
 *
 *  NOTE: Since Worker2 workers are standard node, there are point-to-point
 *        connections between Worker1 and Worker2 (the cardinality must
 *        be the same).
 */

/* Author: Massimo Torquati
 *
 */

#include <iostream>

#include <ff/ff.hpp>

using namespace ff;

struct Emitter: ff_monode_t<long> {
    Emitter(int nworkers, size_t ntasks):nworkers1(nworkers),ntasks(ntasks) {}
    long* svc(long* in) {
        if (in==nullptr) {
            for(int i=0;i<nworkers1 && ntasks>0;++i) {
                ff_send_out_to((long*)ntasks--, i % nworkers1);
            }
            return GO_ON;
        }
        assert(get_channel_id() >= 0 && get_channel_id() < nworkers1);
        ff_send_out_to((long*)ntasks--, get_channel_id());
        if (ntasks==0) {
            return EOS;
        }
        return GO_ON;
    }
    int nworkers1;
    size_t ntasks;
};


struct Worker1: ff_monode_t<long> {
    long *svc(long *in) {
        ff_send_out_to(in, 1);     // forward
        ff_send_out_to(in, 0);     // backward 
        return GO_ON;
    }
};

struct Worker2: ff_node_t<long> {    
    long* svc(long* in) {
        return in;
    }    
};

struct Worker3: ff_node_t<long> {    
    long* svc(long* in) {
        return in;
    }
};


struct Last: ff_minode_t<long> {
    long* svc(long* in) {
        std::cout << "Last: received " << (long)in << "\n";
        return GO_ON;
    }
};

int main() {
    const int nworkers1=3;
    const int nworkers3=2;
    const size_t ntasks=1000;

    // ---- farm without collector
    Emitter E(nworkers1, ntasks);
    std::vector<ff_node*> W;
    for(int i=0;i<nworkers1;++i)
        W.push_back(new Worker1);
    ff_farm farm;
    farm.add_workers(W);
    farm.add_emitter(&E);
    farm.cleanup_workers();
    farm.wrap_around();

    // ----- building all-to-all
    std::vector<ff_node*> firstSet;
    for(int i=0;i<nworkers1;++i)
        firstSet.push_back(new Worker2);
    std::vector<ff_node*> secondSet;
    for(int i=0;i<nworkers3;++i)
        secondSet.push_back(new Worker3);
    
    ff_a2a a2a;
    a2a.add_firstset(firstSet,0, true);
    a2a.add_secondset(secondSet, true);

    // ---- last stage
    Last last;

    // ---- building the topology
    ff_Pipe<> pipe(farm, a2a, last);

    if (pipe.run_and_wait_end() <0) {
        error("running pipe\n");
        return -1;
    }
    return 0;
}
