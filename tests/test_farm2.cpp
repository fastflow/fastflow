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
// farm with emitter and collector explicitly defined as multi-output and
// multi-input, respectively
/*
 *  
 *               |--> Worker -->|  
 *               |              |  
 *   Emitter --> |--> Worker -->|--> Collector
 *               |              |  
 *               |--> Worker -->|  
 *
 * Emitter can be either a standard ff_node or a ff_monode
 * Collector can be either a standard ff_node or a ff_minode
 *
 * The same topology can be constructed by using 2 building blocks:
 *  pipe(Emitter, A2A(Worker, Collector)) 
 *
 */
/* Author: Massimo Torquati
 *
 */
#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

static int WORKTIME_TICKS=25000;

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
        ticks_wait(WORKTIME_TICKS); 
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
        if (argc!=4) {
            std::cerr << "use: " << argv[0] << " [nworkers ntasks ticks]\n";
            return -1;
        }
        nworkers= atol(argv[1]);
        ntasks  = atol(argv[2]);
        WORKTIME_TICKS=atol(argv[3]);
    }

#if 0
    unsigned long start=getusec();
    Worker w;
    for(long i=1;i<=ntasks;++i) 
        w.svc((long*)i);
    unsigned long stop=getusec();
    printf("seq Time=%g (ms)\n", (stop-start)/1000.0);
    {
        Emitter E(nworkers,ntasks);
        Collector C(ntasks);
        Worker W;

        auto comp=combine_nodes(E, combine_nodes(W,C));
        ff_Pipe<> pipe(comp);
        pipe.run();
        pipe.wait();
        printf("comp Time = %g (%g) ms\n", pipe.ffwTime(), pipe.ffTime());
    }
    
#endif

    Emitter E(nworkers,ntasks);
    Collector C(ntasks);
    std::vector<std::unique_ptr<ff_node>> W;
    for(int i=0;i<nworkers;++i) 
        W.push_back(make_unique<Worker>());
    
    ff_Farm<> farm(std::move(W), E, C);
#if 1
    OptLevel1 opt;
    opt.max_nb_threads=ff_realNumCores();
    opt.max_mapped_threads=opt.max_nb_threads;
    opt.no_default_mapping=true;
    opt.verbose_level=2;
    farm.set_scheduling_ondemand();
    
    if (optimize_static(farm,opt)<0) {
        error("optimizing farm\n");
        return -1;
    }
    if (farm.run()<0) {
        error("running farm\n");
        return -1;
    }
    if (farm.wait_collector()<0) {
        error("waiting termination\n");
        return -1;
    }
    printf("farm Time = %g (%g) ms\n", farm.ffwTime(), farm.ffTime());
#else
    // farm with building blocks.......
    ff_a2a a2a;
    std::vector<ff_node*> _W1;
    for(int i=0;i<nworkers;++i)
        _W1.push_back(new Worker);

    a2a.add_firstset(_W1, 1, true);
    std::vector<ff_node*> _W2;
    _W2.push_back(&C);
    a2a.add_secondset(_W2);
    ff_Pipe<> pipe(E, a2a);
    if (pipe.run()<0) {
        error("running pipe\n");
        return -1;
    }
    if (pipe.wait_last()<0) {
        error("waiting termination\n");
        return -1;
    }
    printf("pipe Time = %g (%g) ms\n", pipe.ffwTime(), pipe.ffTime());
#endif    

    return 0;
}

