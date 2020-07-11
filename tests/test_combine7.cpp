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
 * pipeline(farm1, farm2)
 *  
 *         |--> W1 -->|   |<-- comp -->|    
 *         |          |                   | --> W2 --> |
 *  E1 --> |--> W1 -->|--> C1 ----> E2 -->|            | --> C2
 *         |          |                   | --> W2 --> |  
 *         |--> W1 -->|  
 *
 *
 */
/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

struct Emitter1: ff_monode_t<long> { 
    Emitter1(int nworkers, long ntasks):nworkers(nworkers),ntasks(ntasks) {}
    long *svc(long*) {
        for(long i=1;i<=ntasks;++i) 
            ff_send_out_to((long*)i, i % nworkers);
        return EOS;
    }
    int nworkers;
    long ntasks;
};
struct Worker1: ff_node_t<long> {
    long *svc(long *in) {
        //printf("Worker1 (%ld) received %ld\n", get_my_id(), (long)in);
        return in;
    }
};
struct Worker2: ff_node_t<long> {
    long *svc(long *in) {
        printf("Worker2 received %ld\n", (long)in);
        return in;
    }
};
// it can be also an ff_minode_t
struct Collector1: ff_minode_t<long> {
    long *svc(long *in) {
        //printf("Collector1 receved %ld\n", (long)in);
        return in;
    }    
};
// it can be also an ff_monode_t
struct Emitter2: ff_node_t<long> {
    long *svc(long *in) {
        //printf("Emitter2 received %ld\n", (long)in);
        return in;
    }    
};

struct Collector2: ff_minode_t<long> {
    Collector2(long ntasks):ntasks(ntasks) {}
    
    long *svc(long *in) {
        --ntasks;
        std::cout << "Collector received " << (long)in << " from " << get_channel_id() << "\n";
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

int main() {
    int nworkers1 = 2;
    int nworkers2 = 3;
    int ntasks   = 150;


    std::vector<std::unique_ptr<ff_node> > W1;
    for(int i=0;i<nworkers1;++i) 
        W1.push_back(make_unique<Worker1>());

    Emitter1 E(nworkers1,ntasks);
    Collector1 C1;
    ff_Farm<> farm1(std::move(W1), E, C1); 

    std::vector<std::unique_ptr<ff_node> > W2;
    for(int i=0;i<nworkers2;++i) 
        W2.push_back(make_unique<Worker2>());

    Emitter2 E2;
    Collector2 C2(ntasks);
    
    ff_Farm<> farm2(std::move(W2), E2, C2);

    auto pipe = combine_farms(farm1, &C1, farm2,&E2, true);
    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }
    
    printf("DONE\n");
    return 0;
}

