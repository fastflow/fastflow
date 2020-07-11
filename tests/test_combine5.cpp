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
 *
 *  combine(farm1(E,Worker),farm2(E2, Worker1, C)) -------
 *                         ______________________________|      
 *                        | 
 *                        |
 *                         ----> farm(E, A2A(comp(Worker, E2), Worker1), C)
 *
 *
 *            |<------------------- A2A --------------------->| 
 *                   |<------ comp ------> | 
 *
 *               |--> Worker --> Emitter2 -->|  
 *               |                           |--> Worker1-->|  
 *   Emitter --> |--> Worker --> Emitter2 -->|              |  --> Collector
 *               |                           |--> Worker1-->|  
 *               |--> Worker --> Emitter2 -->|  
 *
 * NOTE: Emitter2 must be stateless!!!
 *
 *
 */
/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

struct Emitter: ff_monode_t<long> { 
    Emitter(int nworkers, long ntasks):nworkers(nworkers),ntasks(ntasks) {}
    long *svc(long*) {
        for(long i=1;i<=ntasks;++i) 
            ff_send_out_to((long*)i, i % nworkers);
        return EOS;
    }
    int nworkers;
    long ntasks;
};
struct Worker: ff_node_t<long> {
    long *svc(long *in) {
        return in;
    }
};
// NOTE: Emitter2 can also be a multi-output node
struct Emitter2: ff_node_t<long> {
    long *svc(long* in) {
        ff_send_out(in);
        return GO_ON;
    }
};
struct Worker1: ff_node_t<long> {
    long *svc(long *in) {
        return in;
    }
};

struct Collector: ff_minode_t<long> {
    Collector(long ntasks):ntasks(ntasks) {}
    long *svc(long *) {
        --ntasks;
        //std::cout << "Collector received input from " << get_channel_id() << "\n";
        return GO_ON;
    }
    void svc_end() {
        if (ntasks != 0) {
            std::cerr << "ERROR in the test\n" << "\n";
            std::cerr << "svc_end ntasks= " << ntasks << "\n";
            abort();
        }
    }
    long ntasks;
};

int main(int argc, char* argv[]) {
    int nworkers1 = 1;
    int nworkers2 = 2;
    int ntasks   = 100;
    if (argc>1) {
        if (argc!=4) {
            std::cerr << "use: " << argv[0] << " [nworkers1 nworkers2 ntasks]\n";
            return -1;
        }
        nworkers1= atol(argv[1]);
        nworkers2= atol(argv[2]);
        ntasks   = atol(argv[3]);
    }

    Emitter E(nworkers1,ntasks);

    std::vector<std::unique_ptr<ff_node>> W1;
    for(int i=0;i<nworkers1;++i) 
        W1.push_back(make_unique<Worker>());
    
    ff_Farm<> farm1(std::move(W1), E);  // default collector

    W1.clear();
    for(int i=0;i<nworkers2;++i) 
        W1.push_back(make_unique<Worker1>());

    Emitter2 E2;
    ff_Farm<> farm2(std::move(W1), E2);
    Collector C(ntasks);
    farm2.add_collector(C);

    auto farm = combine_farms_a2a(farm1, E2, farm2);
    unsigned long start = getusec();
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    unsigned long stop = getusec();
    printf("Farm time = %g (ms). Total Time = %g (ms)\n", farm.ffTime(), (stop-start)/1000.0);
    printf("DONE\n");
    return 0;
}

