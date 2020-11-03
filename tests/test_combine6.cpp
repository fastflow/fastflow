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
 * Single farm topology where the emitter and the collector are a composition of nodes
 *  
 *   |<------------ comp ----------->| |--> Worker -->|    |<----------- comp --------->|
 *                                     |              |  
 *   Generator -->Filter1 --->Emit --> |--> Worker -->|--> Coll --> Filter2 --> Gatherer
 *                                     |              |  
 *                                     |--> Worker -->|  
 *
 * NOTE: Since the Emit node is the last stage of a combine, which is added as farm Emitter 
 *       filter, it must be defined as multi-output node.
 * 
 */
/* Author: Massimo Torquati
 *
 */
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

struct Generator: ff_node_t<long> {
    Generator(int ntasks):ntasks(ntasks) {}
    long *svc(long*) {
        for(long i=1;i<=ntasks;++i) 
            ff_send_out((long*)i);
        return EOS;
    }
    int ntasks;
};

struct Emit: ff_monode_t<long> {
    enum { HowMany=1 };
    long *svc(long* in) {
        for(long i=0;i<HowMany;++i) {
            ff_send_out(in);
        }
        return GO_ON;
    }

    void eosnotify(ssize_t ) {
        std::cout << "Emitter eosnotify EOS received\n";
    }

    
};
struct Filter1: ff_node_t<long> {
    long *svc(long *in) {
        return in;
    }
    void eosnotify(ssize_t ) {
        std::cout << "Filter1 EOS received\n";
    }
};
struct Filter2: ff_node_t<long> {
    long *svc(long *in) {
        return in;
    }
    void eosnotify(ssize_t ) {
        std::cout << "Filter2 EOS received\n";
    }


};
struct Worker: ff_node_t<long> {
    long *svc(long *in) {
        if (get_my_id() == 0) usleep(10000);
        return in;
    }

    void eosnotify(ssize_t ) {
        std::cout << "Worker" << get_my_id() << " eosnotify EOS received\n";
    }

    
};
struct Coll: ff_node_t<long> {
    long *svc(long *in) {
        return in;
    }

    void eosnotify(ssize_t ) {
        std::cout << "Collector EOS received\n";
    }
    
};
struct Gatherer: ff_node_t<long> {
    Gatherer(long ntasks):ntasks(ntasks) {}
    long *svc(long *in) {
        --ntasks;
        std::cout << "Gatherer received "<< (long)in << "\n";
        return GO_ON;
    }
    void svc_end() {
        if (ntasks != 0) {
            std::cerr << "ERROR in the test ntasks= " << ntasks << "\n";
            abort();
        }
    }
    long ntasks;
};



int main(int argc, char* argv[]) {
    int nworkers = 4;
    int ntasks   = 1000;
    if (argc>1) {
        if (argc!=3) {
            std::cerr << "use: " << argv[0] << " [nworkers ntasks]\n";
            return -1;
        }
        nworkers= atol(argv[1]);
        ntasks  = atol(argv[2]);
    }

    Generator Gen(ntasks);
    Filter1 Fil1;
    Filter2 Fil2;    
    Emit E;
    Coll C;
    std::vector<std::unique_ptr<ff_node>> W;
    for(int i=0;i<nworkers;++i) 
        W.push_back(make_unique<Worker>());    
    Gatherer Sink(Emit::HowMany * ntasks);

    auto First  = combine_nodes(Gen, combine_nodes(Fil1, E));
    auto Third  = combine_nodes(combine_nodes(C,Fil2), Sink);

    ff_Farm<> farm(std::move(W), First, Third);
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    printf("TEST DONE\n");
    return 0;
}

