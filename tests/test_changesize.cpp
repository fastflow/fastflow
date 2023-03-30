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
 *               |--> Worker0 -->|  
 *               |               |  
 *   Emitter --> |--> Worker1 -->|--> Collector
 *               |               |  
 *               |    ......     |
 *               |--> WorkerK -->|  
 *
 * This program tests the change_outputqueuesize() method when called in the svc_init.
 * Only the Worker0 has both input and output queues set to size=1, the other
 * Workers use the default queue length as set in the config file.
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

    int svc_init() {
        //
        // This is the preferred way to change the input queue capacity
        // of the next node.
        // This action cannot be done directly by the worker because, at
        // the time of execution of the change_inputqueuesize in the worker,
        // the previous node (i.e., this node) might have
        // already pushed some data into the queue we want to modify.
        //
        svector<ff_node*> w;
        this->get_out_nodes(w);
        size_t oldsz;
        w[0]->change_inputqueuesize(1, oldsz);   
        return 0;
    }


    long *svc(long*) {
        for(long i=1;i<=ntasks;++i)
            ff_send_out_to((long*)i, i % nworkers);
        return EOS;
    }
    int nworkers;
    long ntasks;
};
struct Worker: ff_node_t<long> {
    int svc_init() {
        if (get_my_id() == 0) {
#if 0            
            size_t oldsz;
            this->change_outputqueuesize(1,oldsz);
#else
            // this is the preferred way to change the output queue capacity
            svector<ff_node*> w;
            this->get_out_nodes(w);
            assert(w.size() == 1);
            size_t oldsz;
            w[0]->change_outputqueuesize(1, oldsz);                          
#endif
        }
        return 0;
    }
    

    long *svc(long *in) {
        return in;
    }
};
struct Collector: ff_minode_t<long> {
    Collector(long ntasks):ntasks(ntasks) {}
    long *svc(long *) {
        --ntasks;
        return GO_ON;
    }
    void svc_end() {
        if (ntasks != 0) {
            std::cerr << "ERROR in the test\n" << "\n";
            abort();
        }
    }
    long ntasks;
};

int main(int argc, char* argv[]) {
    int nworkers = 4;
    int ntasks   = 10000;
    if (argc>1) {
        if (argc!=3) {
            std::cerr << "use: " << argv[0] << " [nworkers ntasks]\n";
            return -1;
        }
        nworkers= atol(argv[1]);
        ntasks  = atol(argv[2]);
    }


    Emitter E(nworkers,ntasks);
    Collector C(ntasks);
    std::vector<std::unique_ptr<ff_node>> W;
    for(int i=0;i<nworkers;++i) 
        W.push_back(make_unique<Worker>());
    
    ff_Farm<> farm(std::move(W), E, C);
    if (farm.run_and_wait_end()== -1) {
        error("running farm");
        return -1;
    }
    fprintf(stdout, "OK!\n");
    return 0;
}

