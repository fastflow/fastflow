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
 * Testing optimization: introducing normal form and all-to-all.
 *
 *  Original network: 
 *    pipe(First, farm1(W1), farm2(W2), farm3(W3), farm1(W1), farm2(W2), farm3(W3), Last) 
 * 
 *  After (automagic) optimizations:
 *    pipe(farm(First, A2A(comb(W1,W2,W3,W4,W5,W6)), Last) )
 *
 *
 *    |<------------------------ farm ----------------------->|
 *            |<----------------- A2A ------------------>|  
 *                 |<--- comp --->|     |<-- comp -->|
 *
 *              | --> W1+W2+W3+W4 -->|
 *              |                    | ---> W5+W6 ---> |
 *     First -->| --> W1+W2+W3+W4 -->|                 | --> Last
 *              |                    | ---> W5+W6 ---> |
 *              | --> W1+W2+W3+W4 -->|
 *
 *
 * NOTE: After the optimize_static call, the number of threads is considerably reduced.
 *
 *
 */
/* Author: Massimo Torquati
 *
 */

#include <string>
#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

struct First: ff_node_t<long> {
    First(const int ntasks):ntasks(ntasks) {}
    long* svc(long*) {
        for(long i=1;i<=ntasks;++i) {
            struct timespec req;
            req.tv_sec = 0;
            req.tv_nsec = 5000;
            nanosleep(&req, (struct timespec *)NULL);

            ff_send_out((long*)i);
        }
        return EOS;
    }
    const int ntasks;
};
struct Stage1: ff_node_t<long> {
    long* svc(long*in) {
        //printf("Stage1 (%ld) %ld\n", get_my_id(), (long)in);
        struct timespec req;
        req.tv_sec = 0;
        req.tv_nsec = 30000;
        nanosleep(&req, (struct timespec *)NULL);
        
        return in;
    }
};
struct Stage2: ff_node_t<long> {
    long* svc(long*in) {
        //printf("Stage2 (%ld) %ld\n", get_my_id(), (long)in);
        struct timespec req;
        req.tv_sec = 0;
        req.tv_nsec = 30000;
        nanosleep(&req, (struct timespec *)NULL);

        return in;
    }
};
struct Stage3: ff_node_t<long> {
    long* svc(long*in) {
        //printf("Stage3 (%ld) %ld\n", get_my_id(), (long)in);
        struct timespec req;
        req.tv_sec = 0;
        req.tv_nsec = 30000;
        nanosleep(&req, (struct timespec *)NULL);
        
        return in;
    }
};
struct Stage4: ff_node_t<long> {
    long* svc(long*in) {
        //printf("Stage4 (%ld) %ld\n", get_my_id(), (long)in);
        struct timespec req;
        req.tv_sec = 0;
        req.tv_nsec = 30000;
        nanosleep(&req, (struct timespec *)NULL);

        return in;
    }
};
struct Stage5: ff_node_t<long> {
    long* svc(long*in) {
        //printf("Stage5 (%ld) %ld\n", get_my_id(), (long)in);
        struct timespec req;
        req.tv_sec = 0;
        req.tv_nsec = 20000;
        nanosleep(&req, (struct timespec *)NULL);

        return in;
    }
};
struct Stage6: ff_node_t<long> {
    long* svc(long*in) {
        //printf("Stage6 (%ld) %ld\n", get_my_id(), (long)in);
        struct timespec req;
        req.tv_sec = 0;
        req.tv_nsec = 20000;
        nanosleep(&req, (struct timespec *)NULL);
        
        return in;
    }    
};
struct Last: ff_node_t<long> {
    long* svc(long*) {
        //printf("received %ld\n", (long)in);
        ++counter;
        return GO_ON;
    }
    size_t counter=0;
};


int main(int argc, char* argv[]) {

    // default arguments
    size_t ntasks    = 10000;
    bool   optimize  = true;
    size_t nworkers1 = 3;
    size_t nworkers2 = 2;


    if (argc>1) {
        if (argc!=5) {
            error("use: %s ntasks nworkers1 nworkers2 optimize\n",argv[0]);
            return -1;
        }
        ntasks    = std::stol(argv[1]);
        nworkers1 = std::stol(argv[2]);
        nworkers2 = std::stol(argv[3]);
        optimize  = (std::stol(argv[4])!=0);
    }

    First first(ntasks);
    Last last;


    /* all the following farms have nworkers1 workers */
    ff_Farm<long,long> farm1([&]() {
	    std::vector<std::unique_ptr<ff_node> > V;
	    for(size_t i=0;i<nworkers1;++i)
		V.push_back(make_unique<Stage1>());
	    return V;
	} ());
    ff_Farm<long,long> farm2([&]() {
	    std::vector<std::unique_ptr<ff_node> > V;
	    for(size_t i=0;i<nworkers1;++i)
		V.push_back(make_unique<Stage2>());
	    return V;
	} ());


    ff_Farm<long,long> farm3([&]() {
	    std::vector<std::unique_ptr<ff_node> > V;
	    for(size_t i=0;i<nworkers1;++i)
		V.push_back(make_unique<Stage3>());
	    return V;
	} ());
    ff_Farm<long,long> farm4([&]() {
	    std::vector<std::unique_ptr<ff_node> > V;
	    for(size_t i=0;i<nworkers1;++i)
		V.push_back(make_unique<Stage4>());
	    return V;
	} ());
    /* ---------------------------------------------------- */

    /* the following farms have nworkers2 workers */
    ff_Farm<long,long> farm5([&]() {
	    std::vector<std::unique_ptr<ff_node> > V;
	    for(size_t i=0;i<nworkers2;++i)
		V.push_back(make_unique<Stage5>());
	    return V;
	} ());

    ff_Farm<long,long> farm6([&]() {
	    std::vector<std::unique_ptr<ff_node> > V;
	    for(size_t i=0;i<nworkers2;++i)
		V.push_back(make_unique<Stage6>());
	    return V;
	} ());
    /* ---------------------------------------------------- */

#if 0
    farm1.set_scheduling_ondemand();
    farm2.set_scheduling_ondemand();
    farm3.set_scheduling_ondemand();
    farm4.set_scheduling_ondemand();
    farm5.set_scheduling_ondemand();
    farm6.set_scheduling_ondemand();
#endif
    // original network
    //ff_Pipe<> pipe(first, farm1, farm2, farm3, farm4, farm5, farm6, last);
    ff_Pipe<> pipe(first, farm1, farm2, farm3, farm4, farm5, farm6, last);

    // optimization that I would like to apply, if possible
    if (optimize) {
        OptLevel opt;
        opt.max_nb_threads=ff_realNumCores();
        opt.max_mapped_threads=opt.max_nb_threads;
        opt.verbose_level=2;
        opt.no_initial_barrier=true;
        opt.no_default_mapping=true; // disable mapping if #threads > max_mapped_threads
        opt.blocking_mode=true;      // enabling blocking if #threads > max_nb_threads
        opt.merge_farms=true;        // introducing normal form, if possible
        opt.merge_with_emitter=true; // merging previous pipeline stage with farm emitter 
        opt.remove_collector=true;   // remove farm collector
        opt.introduce_a2a=true;      // introduce all-2-all between two farms, if possible
        
        // the next call tries to apply all previous optimizations by changing the 
        // internal structure of the pipe passed as parameter
        if (optimize_static(pipe,opt)<0) {
            error("optimize_static\n");
            return -1;
         }
    }
    printf("the n. of threads implementing the pipe is %d\n", pipe.cardinality());
    // running the optimized pipe
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    printf("Time = %g (ms)\n",pipe.ffTime());
    if (last.counter != ntasks) {
        printf("test FAILED (%ld)\n", last.counter);
        return -1;
    }
    printf("test DONE\n");
    return 0;
}
