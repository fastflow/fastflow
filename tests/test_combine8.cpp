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
 * Transforming a pipeline of two farms. 
 * This test stress the combine_farms(farm1,node1,farm2,node2,bool) function.
 *
 *  TEST1 is as follows: 
 * 
 *  pipeline(farm1, farm2) -->   
 *
 * |<--------- farm1 --------->| |<--------- farm2 --------->|
 *
 *         |--> W1 -->|      
 *         |          |                  | --> W2 --> |
 *  E1 --> |--> W1 -->| --> C1 --> E2--> |            | --> C2
 *         |          |                  | --> W2 --> |  
 *         |--> W1 -->|  
 *
 *
 *  --> farm(A2A(comp, comp))  (single farm with a single A2A worker)
 *
 *
 *  |<------------------------ farm ------------------------->|
 *       |<-------------- A2A (farm worker) ------------->| 
 *        
 *           |<-- comp -->|              |<-- comp -->|
 *         |--> W1 -->C1 -->|      
 *         |                |       | --> E2 --> W2 --> |
 *  E1 --> |--> W1 -->C1 -->| ----> |                   | --> C2
 *         |                |       | --> E2 --> W2 --> |  
 *         |--> W1 -->C1 -->|  
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
        if (get_my_id()==0 || get_my_id()==1) usleep(10000);
        //printf("Worker1 (%ld) received %ld\n", get_my_id(), (long)in);
        ff_send_out(in);
        return GO_ON;
    }
};
struct Worker2: ff_node_t<long> {
    long *svc(long *in) {
        //printf("Worker2 received %ld\n", (long)in);
        return in;
    }
};
struct Collector1: ff_node_t<long> {
    long *svc(long *in) {
        //printf("Collector1 receved %ld\n", (long)in);
        ff_send_out(in);
        return GO_ON;
    }
};
struct Emitter2: ff_node_t<long> {
    long *svc(long *in) {
        //printf("Emitter2 received %ld\n", (long)in);
        ff_send_out(in);
        return GO_ON;
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

int main(int argc, char* argv[]) {
    int nworkers1 = 2;
    int nworkers2 = 3;
    int ntasks   = 1000;
    if (argc>1) {
        if (argc!=4) {
            error("use: %s nworkers1 nworkers2 ntasks\n", argv[0]);
            return -1;
        }
        nworkers1=atoi(argv[1]);
        nworkers2=atoi(argv[2]);
        ntasks=atoi(argv[3]);
    }
    {
        // it produces a farm of all-to-all whose nodes
        // are compositions of the two farm workers
        
        Emitter1 E(nworkers1,ntasks);
        Collector1 C1;
        Emitter2 E2;
        Collector2 C2(ntasks);
        std::vector<std::unique_ptr<ff_node> > W1;
        for(int i=0;i<nworkers1;++i) 
            W1.push_back(make_unique<Worker1>());
        std::vector<std::unique_ptr<ff_node> > W2;
        for(int i=0;i<nworkers2;++i) 
            W2.push_back(make_unique<Worker2>());
        ff_Farm<> farm1(std::move(W1), E, C1); 
        ff_Farm<> farm2(std::move(W2), E2, C2);        
        
        auto pipe = combine_farms(farm1, &C1, farm2, &E2, false);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }    
        printf("TEST1 DONE\n");
    }
    usleep(500000);
    {
        // The request is to merge the 2 farms. ince nworkers1!=nworkers2
        // it produces a farm of all-to-all
        
        Emitter1 E(nworkers1,ntasks);
        Collector1 C1;
        Emitter2 E2;
        Collector2 C2(ntasks);
        std::vector<std::unique_ptr<ff_node> > W1;
        for(int i=0;i<nworkers1;++i) 
            W1.push_back(make_unique<Worker1>());
        std::vector<std::unique_ptr<ff_node> > W2;
        for(int i=0;i<nworkers2;++i) 
            W2.push_back(make_unique<Worker2>());
        ff_Farm<> farm1(std::move(W1), E, C1); 
        ff_Farm<> farm2(std::move(W2), E2, C2);        

        auto pipe = combine_farms(farm1, (Collector1*)nullptr, farm2, (Collector1*)nullptr, true);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }    
        printf("TEST2 DONE\n");
    }
    usleep(500000);
    {
        // merging C1 with E2 
        
        Emitter1 E(nworkers1,ntasks);
        Collector1 C1;
        Emitter2 E2;
        Collector2 C2(ntasks);
        std::vector<std::unique_ptr<ff_node> > W1;
        for(int i=0;i<nworkers1;++i) 
            W1.push_back(make_unique<Worker1>());
        std::vector<std::unique_ptr<ff_node> > W2;
        for(int i=0;i<nworkers2;++i) 
            W2.push_back(make_unique<Worker2>());
        ff_Farm<> farm1(std::move(W1), E, C1); 
        ff_Farm<> farm2(std::move(W2), E2, C2);        

        auto pipe = combine_farms(farm1, &C1, farm2, &E2, true);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }    
        printf("TEST3 DONE\n");
    }
    usleep(500000);
    {
        // sets C1 as emitter of the second farm (C1 is multi-input)
        // the collector of the first farm is removed

        Emitter1 E(nworkers1,ntasks);
        struct Collector1: ff_minode_t<long> {   // multi-input node
            long *svc(long *in) {
                printf("Collector1 receved %ld\n", (long)in);
                return in;
            }
        } C1;

        Emitter2 E2;
        Collector2 C2(ntasks);
        std::vector<std::unique_ptr<ff_node> > W1;
        for(int i=0;i<nworkers1;++i) 
            W1.push_back(make_unique<Worker1>());
        std::vector<std::unique_ptr<ff_node> > W2;
        for(int i=0;i<nworkers2;++i) 
            W2.push_back(make_unique<Worker2>());
        ff_Farm<> farm1(std::move(W1), E, C1); 
        ff_Farm<> farm2(std::move(W2), E2, C2);        

        auto pipe = combine_farms(farm1, &C1, farm2, (Collector1*)nullptr, true);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }    
        printf("TEST4 DONE\n");
    }
    usleep(500000);
    {
        // similar to the previous test. In this case C1 executes an 
        // all_gather
        
        Emitter1 E(nworkers1,ntasks);
        struct Collector1: ff_minode_t<long> {
            long *svc(long *in) {
                printf("Collector1 received=%ld\n", (long)in);
                std::vector<long*> V;
                all_gather(in,V);
                for(size_t i=0;i<V.size();++i) {
                    if (V[i]) ff_send_out(V[i]);
                }
                return GO_ON;
            }
        } C1;
        
        Emitter2 E2;
        Collector2 C2(ntasks);
        std::vector<std::unique_ptr<ff_node> > W1;
        for(int i=0;i<nworkers1;++i) 
            W1.push_back(make_unique<Worker1>());
        std::vector<std::unique_ptr<ff_node> > W2;
        for(int i=0;i<nworkers2;++i) 
            W2.push_back(make_unique<Worker2>());
        ff_Farm<> farm1(std::move(W1), E, C1); 
        ff_Farm<> farm2(std::move(W2), E2, C2);        

        auto pipe = combine_farms(farm1, &C1, farm2, &E2, true);        
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }    
        printf("TEST4bis DONE\n");
    }
    usleep(500000);
    {
        Emitter1 E(nworkers1,ntasks);
        Collector1 C1;
        struct Emitter2: ff_minode_t<long> {
            long *svc(long *in) {
                printf("Emitter2 received %ld from %ld\n", (long)in, get_channel_id());
                return in;
            }    
        } E2;

        Collector2 C2(ntasks);
        std::vector<std::unique_ptr<ff_node> > W1;
        for(int i=0;i<nworkers1;++i) 
            W1.push_back(make_unique<Worker1>());
        std::vector<std::unique_ptr<ff_node> > W2;
        for(int i=0;i<nworkers2;++i) 
            W2.push_back(make_unique<Worker2>());
        ff_Farm<> farm1(std::move(W1), E, C1); 
        ff_Farm<> farm2(std::move(W2), E2, C2);        

        auto pipe = combine_farms(farm1, (Collector1*)nullptr, farm2, &E2, true);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }    
        printf("TEST5 DONE\n");
    }    
    usleep(500000);
    {
        Emitter1 E(nworkers1,ntasks);
        Collector1 C1;
        Emitter2 E2;
        Collector2 C2(ntasks);
        std::vector<std::unique_ptr<ff_node> > W1;
        for(int i=0;i<nworkers1;++i) 
            W1.push_back(make_unique<Worker1>());
        std::vector<std::unique_ptr<ff_node> > W2;
        for(int i=0;i<nworkers2;++i) 
            W2.push_back(make_unique<Worker2>());
        ff_Farm<> farm1(std::move(W1), E, C1); 
        ff_Farm<> farm2(std::move(W2), E2, C2);        

        auto pipe = combine_farms(farm1, (Collector1*)nullptr, farm2, (Collector1*)nullptr, false);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }    
        printf("TEST6 DONE\n");
    }
    usleep(500000);
    {
        // here the two farms have the same number of workers
        Emitter1 E(nworkers1,ntasks);
        Collector1 C1;
        Emitter2 E2;
        Collector2 C2(ntasks);
        std::vector<std::unique_ptr<ff_node> > W1;
        for(int i=0;i<nworkers1;++i) 
            W1.push_back(make_unique<Worker1>());
        std::vector<std::unique_ptr<ff_node> > W2;
        for(int i=0;i<nworkers1;++i) 
            W2.push_back(make_unique<Worker2>());
        ff_Farm<> farm1(std::move(W1), E, C1); 
        ff_Farm<> farm2(std::move(W2), E2, C2);        

        auto pipe = combine_farms(farm1, (Collector1*)nullptr, farm2, (Collector1*)nullptr, true);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }    
        printf("TEST7 DONE\n");
    }
    usleep(500000);
    {
        Emitter1 E(nworkers1,ntasks);
        Collector1 C1;
        struct Emitter2: ff_monode_t<long> {  // <-- multi-output
            long *svc(long *in) {
                printf("Emitter2 receved %ld\n", (long)in);
                printf("Emitter2 sending to worker%ld\n", idx);
                ff_send_out_to(in, idx);
                ++idx  %= get_num_outchannels();
                return GO_ON;
            }
            long idx=0;
        } E2;

        Collector2 C2(ntasks);
        std::vector<std::unique_ptr<ff_node> > W1;
        for(int i=0;i<nworkers1;++i) 
            W1.push_back(make_unique<Worker1>());
        std::vector<std::unique_ptr<ff_node> > W2;
        for(int i=0;i<nworkers2;++i) 
            W2.push_back(make_unique<Worker2>());
        ff_Farm<> farm1(std::move(W1), E, C1); 
        ff_Farm<> farm2(std::move(W2), E2, C2);        

        auto pipe = combine_farms(farm1, &C1, farm2, &E2, true);        
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }    
        printf("TEST8 DONE\n");
    }
}

