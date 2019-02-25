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
 * testing the helper function 'combine_nodes_in_pipeline' together with eosnotify
 *
 */
/* Author: Massimo Torquati
 *
 */
#include <ff/ff.hpp>

using namespace ff;

size_t ntasks=1111;

struct Generator: ff_monode_t<long> {
    long *svc(long *) {
	for(size_t i=1;i<=ntasks;++i)	    
	    ff_send_out((long*)i);
        return EOS;
    }
};

struct Emitter: ff_monode_t<long> {
    long *svc(long *in) {
        ff_send_out(in);
        return GO_ON;
    }
};
struct Worker: ff_node_t<long> {
    long* svc(long*in) {
        return in;
    }
};
struct Collector: ff_node_t<long> {   // ff_minode_t<long> {
    Collector(size_t nworkers):nworkers(nworkers) {}
    long *svc(long *in) {
        return in;
    }
    void eosnotify(ssize_t=-1) {
        //if (++neos == nworkers) ff_send_out(EOS); // redundant
    }

    size_t neos=0;
    size_t nworkers;
};
struct Seq: ff_node_t<long> {
    long *svc(long *in) {
        V.push_back(in);
        if (V.size() == soglia) {
            for(size_t i=0;i<V.size(); ++i)
                ff_send_out(V[i]);
            V.resize(0);
        } 
        return GO_ON;
    }
    void eosnotify(ssize_t=-1) {
        for(size_t i=0;i<V.size(); ++i)
            ff_send_out(V[i]);
        V.resize(0);
    }    
    std::vector<long*> V;
    const size_t soglia=10;
};
struct Gatherer: ff_node_t<long> {
    long *svc(long *in) {
        printf("received %ld (%ld)\n", (long)in, ++cnt);
        return GO_ON;
    }
    void svc_end() {
        if (ntasks!=cnt) {
            error("TEST FAILED\n");
            exit(-1);
        }
    }
    size_t cnt=0;
};


int main(int argc, char* argv[]) {
    if (argc==2) {
        ntasks=atol(argv[1]);
    }
    {
        Generator gen;      
        Gatherer  last;
        
        size_t nworkers=3;
        // first farm, everything allocated on the heap
        ff_farm* farm1 = new ff_farm;	
        farm1->add_emitter(new Emitter);
        farm1->add_collector(new Collector(nworkers));
        std::vector<ff_node*> W1;
        for(size_t i=0;i<nworkers;++i)
            W1.push_back(new Worker);
        farm1->add_workers(W1);
        farm1->cleanup_all();
        
        Seq* seq = new Seq;
        
        auto optpipe = combine_nodes_in_pipeline(*farm1, *seq, true, true);
        ff_pipeline * mypipe = new ff_pipeline;
        mypipe->add_stage(optpipe);
        
        ff_Pipe<> mainpipe(gen, mypipe, last);
        if (mainpipe.run_and_wait_end()<0) {
            error("running farm\n");
            return -1;
        }
    }
    usleep(500000);
    printf("TEST1 DONE\n");
    {
        Generator gen;      
        Gatherer  last;

        size_t nworkers=2;

        // first sequential node, allocated on the heap
        Seq* seq = new Seq;

        // farm, everything allocated on the heap
        ff_farm* farm1 = new ff_farm;	
        farm1->add_emitter(new Emitter);
        farm1->add_collector(new Collector(nworkers));
        std::vector<ff_node*> W1;
        for(size_t i=0;i<nworkers;++i)
            W1.push_back(new Worker);
        farm1->add_workers(W1);
        farm1->cleanup_all();
                
        auto optpipe = combine_nodes_in_pipeline(*seq, *farm1, true, true);
        ff_pipeline * mypipe = new ff_pipeline;
        mypipe->add_stage(optpipe);
        
        ff_Pipe<> mainpipe(gen, mypipe, last);
        if (mainpipe.run_and_wait_end()<0) {
            error("running farm\n");
            return -1;
        }
    }
    usleep(500000);
    printf("TEST2 DONE\n");
    {
        Generator gen;      
        Gatherer  last;
        
        size_t nworkers=2;

        // farm, everything allocated on the heap
        ff_farm* farm0 = new ff_farm;	
        farm0->add_emitter(new Emitter);
        farm0->add_collector(new Collector(nworkers));
        std::vector<ff_node*> W0;
        for(size_t i=0;i<nworkers;++i)
            W0.push_back(new Seq);
        farm0->add_workers(W0);
        farm0->cleanup_all();

        // farm, everything allocated on the heap
        ff_farm* farm1 = new ff_farm;	
        farm1->add_emitter(new Emitter);
        farm1->add_collector(new Collector(nworkers));
        std::vector<ff_node*> W1;
        for(size_t i=0;i<nworkers;++i)
            W1.push_back(new Worker);
        farm1->add_workers(W1);
        farm1->cleanup_all();
                
        auto optpipe = combine_nodes_in_pipeline(*farm0, *farm1, true, true);
        ff_pipeline * mypipe = new ff_pipeline;
        mypipe->add_stage(optpipe);
        
        ff_Pipe<> mainpipe(gen, mypipe, last);
        if (mainpipe.run_and_wait_end()<0) {
            error("running farm\n");
            return -1;
        }
    }
    usleep(500000);
    printf("TEST3 DONE\n");
    return 0;
}
