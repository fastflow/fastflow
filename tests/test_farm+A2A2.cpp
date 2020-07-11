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
/* This example tests the way a farm and an all-to-all connect themselves. 
 *
 *
 * 
 *          |--> Worker ---> |--> First -->|                |--> Third --> Second ->|
 *          |                |             |-->Second ->|   |                       |
 * Source-->|--> Worker ---> |--> First -->|            |-->|--> Third --> Second ->| ----> Collector
 *          |                |             |-->Second ->|   |                       |
 *          |--> Worker ---> |--> First -->|                |--> Third --> Second ->|
 *         
 *                       
 * |<- farm master-worker ->||<------ all-to-all ------>||<------ all-to-all ------>||<- multi-input ->| 
 *        (no collector)                (A2A1)                      (A2A2)
 *
 *
 */
/*
 * Author: Massimo Torquati
 *
 */

#include <ff/ff.hpp>

using namespace ff;

struct Starter: ff_monode_t<long> {
    long* svc(long*) {
	    for(long i=1;i<=ntasks;++i)
            ff_send_out((long*)i);
        return EOS;
    }
    const long ntasks= 10;
}; 
struct WorkerStandard: ff_node_t<long> {
    long* svc(long* in) {
        return in;
    }
};
struct WorkerMO: ff_monode_t<long> {
    long* svc(long* in) {
        return in;
    }
};
struct FirstStandard: ff_node_t<long> {
    long* svc(long* in) {
        return in;
    }
};
struct FirstMI: ff_minode_t<long> {
    long* svc(long* in) {
        return in;
    }
};
struct Second: ff_node_t<long> {
    long* svc(long* in) { return in;}
};
struct SecondMO: ff_monode_t<long> {
    long* svc(long* in) { return in;}
};
struct Third: ff_node_t<long> {
    long* svc(long* in) {
        return in;
    }
};
struct ThirdMI: ff_minode_t<long> {
    long* svc(long* in) {
        return in;
    }
};
struct moHelper: ff_monode_t<long> {
    long* svc(long* in) {
        return in;
    }
};
struct miHelper: ff_minode_t<long> {
    long* svc(long* in) {
        return in;
    }
};
struct Collector: ff_minode_t<long> {
    long* svc(long*) {
	printf("Collector, received from %ld\n", get_channel_id());
	return GO_ON;
    }
};

int main() {
    // #Workers == #L-Workers A2A1, standard Workers
    {
        const int nworkers=3;
        // preparing the master-worker farm
        Starter start;
        std::vector<ff_node*> W;
        for(int i=0;i<nworkers;++i) 
            W.push_back(new WorkerStandard);
        ff_farm farm(W,&start);
        farm.remove_collector();
        farm.cleanup_workers();
        
        // preparing the all-to-all
        ff_a2a a2a;
        std::vector<ff_node*> W1;
        for(int i=0;i<nworkers;++i) {
            W1.push_back(new FirstStandard);
        }
        a2a.add_firstset(W1, 0, true);
        std::vector<ff_node*> W2;
        W2.push_back(new Second);
        W2.push_back(new Second);
        a2a.add_secondset(W2, true);
        
        Collector col;
        ff_Pipe<> pipe(farm, a2a, col);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }
    }
    printf("TEST1 OK\n");
    // #Workers == #L-Workers A2A1, multi-output Workers
    {
        const int nworkers=4;
        // preparing the master-worker farm
        Starter start;
        std::vector<ff_node*> W;
        for(int i=0;i<nworkers;++i) 
            W.push_back(new WorkerMO);
        ff_farm farm(W,&start);
        farm.remove_collector();
        farm.cleanup_workers();
        
        // preparing the all-to-all
        ff_a2a a2a;
        std::vector<ff_node*> W1;
        for(int i=0;i<nworkers;++i) {
            W1.push_back(new FirstStandard);
        }
        a2a.add_firstset(W1, 0, true);
        std::vector<ff_node*> W2;
        W2.push_back(new Second);
        W2.push_back(new Second);
        a2a.add_secondset(W2,true);
        
        Collector col;
        ff_Pipe<> pipe(farm, a2a, col);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }
    }
    printf("TEST2 OK\n");
    // #Workers != #L-Workers A2A1, multi-output Workers and multi-input L-Workers of the A2A1 
    {
        const int nworkers1=4;
        const int nworkers2=2;
        // preparing the master-worker farm
        Starter start;
        std::vector<ff_node*> W;
        for(int i=0;i<nworkers1;++i) 
            W.push_back(new WorkerMO);
        ff_farm farm(W,&start);
        farm.remove_collector();
        farm.cleanup_workers();
        
        // preparing the all-to-all
        ff_a2a a2a;
        std::vector<ff_node*> W1;
        for(int i=0;i<nworkers2;++i) {
            ff_comb *comb = new ff_comb(new FirstMI, new moHelper, true, true);
            W1.push_back(comb);
        }
        a2a.add_firstset(W1, 0, true);
        std::vector<ff_node*> W2;
        W2.push_back(new Second);
        W2.push_back(new Second);
        a2a.add_secondset(W2, true);
        
        Collector col;
        ff_Pipe<> pipe(farm, a2a, col);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }
    }
    printf("TEST3 OK\n");
    // #Workers != #L-Workers A2A1, multi-output Workers and multi-input L-Workers of the A2A1 
    // #R-Workers of the A2A1 == #L-Workers of the A2A2
    {
        const int nworkers1=4;
        const int nworkers2=2;
        // preparing the master-worker farm
        Starter start;
        std::vector<ff_node*> W;
        for(int i=0;i<nworkers1;++i) 
            W.push_back(new WorkerMO);
        ff_farm farm(W,&start);
        farm.remove_collector();
        farm.cleanup_workers();
        
        // preparing the all-to-all
        ff_a2a a2a1;
        std::vector<ff_node*> W1;
        for(int i=0;i<nworkers2;++i) {
            ff_comb *comb = new ff_comb(new FirstMI, new moHelper, true, true);
            W1.push_back(comb);
        }
        a2a1.add_firstset(W1, 0, true);
        std::vector<ff_node*> W2;
        W2.push_back(new Second);
        W2.push_back(new Second);
        a2a1.add_secondset(W2);

        ff_a2a a2a2;
        std::vector<ff_node*> W1_2;
        W1_2.push_back(new Third);
        W1_2.push_back(new Third);
        a2a2.add_firstset(W1_2, 0, true);
        std::vector<ff_node*> W2_2;
        W2_2.push_back(new Second);
        W2_2.push_back(new Second);
        W2_2.push_back(new Second);
        a2a2.add_secondset(W2_2, true);
        
        Collector col;
        ff_Pipe<> pipe(farm, a2a1, a2a2, col);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }
    }
    printf("TEST4 OK\n");
    // #Workers != #L-Workers A2A1, multi-output Workers and multi-input L-Workers of the A2A1 
    // #R-Workers of the A2A1 != #L-Workers of the A2A2
    {
        const int nworkers1=4;
        const int nworkers2=2;
        // preparing the master-worker farm
        Starter start;
        std::vector<ff_node*> W;
        for(int i=0;i<nworkers1;++i) 
            W.push_back(new WorkerMO);
        ff_farm farm(W,&start);
        farm.remove_collector();
        farm.cleanup_workers();
        
        // preparing the all-to-all
        ff_a2a a2a1;
        std::vector<ff_node*> W1;
        for(int i=0;i<nworkers2;++i) {
            ff_comb *comb = new ff_comb(new FirstMI, new moHelper, true, true);
            W1.push_back(comb);
        }
        a2a1.add_firstset(W1, 0, true);
        std::vector<ff_node*> W2;
        W2.push_back(new ff_comb(new miHelper, new SecondMO, true, true));
        W2.push_back(new ff_comb(new miHelper, new SecondMO, true, true));
        a2a1.add_secondset(W2);

        ff_a2a a2a2;
        std::vector<ff_node*> W1_2;
        W1_2.push_back(new ff_comb(new ThirdMI, new moHelper, true, true));
        W1_2.push_back(new ff_comb(new ThirdMI, new moHelper, true, true));
        W1_2.push_back(new ff_comb(new ThirdMI, new moHelper, true, true));
        a2a2.add_firstset(W1_2, 0, true);
        std::vector<ff_node*> W2_2;
        W2_2.push_back(new Second);
        W2_2.push_back(new Second);
        W2_2.push_back(new Second);
        a2a2.add_secondset(W2_2, true);
        
        Collector col;
        ff_Pipe<> pipe(farm, a2a1, a2a2, col);
        if (pipe.run_and_wait_end()<0) {
            error("running pipe\n");
            return -1;
        }
    }
    printf("TEST5 OK\n");
    return 0;
}
