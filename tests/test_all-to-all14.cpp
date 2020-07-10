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

/* Testing a farm as R-Worker of an all-to-all. It tests also the function 
 * combine_with_firststage.
 *
 *                                              
 *                        | --> Emitter(1) -->| 
 *                        |                   |                     |-> First -->|
 *   Gen--> DefEmitter -->|                   | ------> EmitterX -->|            | --> Gat
 *                        |                   |                     |-> First -->|   
 *                        | --> Emitter(2) -->|   (ProxyNode)
 * 
 *                                                 |<-mi emitter->|
 *                                                 |<--- farm with no collector -->|
 *                                              |<---- pipeline ------------------>|
 *                        |<------------------- all-to-all ----------------------->|
 *                        |<------------------- pipeline ------------------------->|
 *        |<---------------------------- farm with no collector ------------------>|
 *
 *
 */

/*
 * Author: Massimo Torquati
 */ 

#include <ff/ff.hpp>

using namespace ff;

struct Generator: ff_node_t<long> {
    Generator(long streamlen):streamlen(streamlen) {}
    long* svc(long*) {
	for(long i=1;i<=streamlen;++i) {
	    ff_send_out((long*)i);
	}
	return EOS;
    }
    long streamlen;
};
struct Gatherer: ff_minode_t<long> {
    long* svc(long*) {
	return GO_ON;
    }
};

struct FarmA2A: ff_pipeline {
    struct Emitter:ff_node_t<long> {  
        long* svc(long* in) {
            return in;
        }    
    };
    struct ProxyNode:ff_minode_t<long> {
        long* svc(long* in) {
            ssize_t id = get_channel_id();
            if (id <0 || id>1) abort();
            //printf("ProxyNode received task from L-Worker %ld\n", id);
            return in;
        }    
    };   
    struct EmitterX:ff_node_t<long> {
        long* svc(long* in) {
            return in;
        }    
    };
    
    struct First: ff_node_t<long> {
        long* svc(long* in) {
            return in;
        }    
    };
    
    FarmA2A() {
        
        ff_a2a* a2a = new ff_a2a;
        std::vector<ff_node*> W1;
        Emitter *e1 = new Emitter;
        Emitter *e2 = new Emitter;
        W1.push_back(e1);
        W1.push_back(e2); 
        a2a->add_firstset(W1, 0, true);
        
        
        ff_pipeline* pipe = new ff_pipeline;
        ff_farm *farm1 = new ff_farm();
        EmitterX *E=new EmitterX;
        farm1->add_emitter(E);
        First* first1 = new First;
        First* first2 = new First;
        std::vector<ff_node*> w;
        w.push_back(first1);
        w.push_back(first2);
        farm1->cleanup_all();
        farm1->add_workers(w);
        //farm1->add_collector(nullptr);
        
        pipe->add_stage(farm1, true);
        
        std::vector<ff_node*> W2;
        W2.push_back(pipe);
        a2a->add_secondset(W2,true);
        
        ff_farm *farm = new ff_farm();
        std::vector<ff_node*> W;
        W.push_back(a2a);
        farm->add_workers(W);
        farm->cleanup_all();	
        
        add_stage(farm, true);

        ProxyNode* pn = new ProxyNode;
        combine_with_firststage(*pipe, pn, true);
    }
};


int main(int argc, char* argv[]) {
    long streamlen = 1000;
    if (argc>1) {
        streamlen = std::stol(argv[1]);
    }
    Generator gen(streamlen);
    Gatherer gat;
    FarmA2A  farma2a;
    ff_Pipe<> pipe(gen, farma2a, gat);

    if (pipe.run_and_wait_end()<0) {
	error("running pipe\n");
	return -1;
    }
    printf("TEST DONE\n");
    return 0;
}
