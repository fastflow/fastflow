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
 * This test checks the optimizations of pipelines inside a farm.
 *
 *   pipe(First, farm(pipe(farm(Worker1), farm(Worker2))), Last)
 *
 *
 * if optimize=true:
 *
 *                                
 *                                |--> Worker1 -->| --> Worker2|
 *             | --> DefEmitter-->|               |            |-->|
 *             |                  |--> Worker1 -->| --> Worker2|   |
 *    First -->|                                                   |-->Last
 *             |                  |--> Worker1 -->| --> Worker2|   |
 *             | --> DefEmitter-->|               |            |-->| 
 *                                |--> Worker1 -->| --> Worker2|
 *   
 *                                 |<---------- A2A ----------->|
 *                   |<--------farm with no collector--- ------>|
 *   |<----------------------- farm with no collector------------->|
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
            req.tv_nsec = 3000;
            nanosleep(&req, (struct timespec *)NULL);

            ff_send_out((long*)i);
        }
        return EOS;
    }
    const int ntasks;
};

struct Worker1: ff_node_t<long> {
    long* svc(long*in) {
        switch(get_my_id()) {
        case 0: {
            struct timespec req;
            req.tv_sec = 0;
            req.tv_nsec = 50000;
            nanosleep(&req, (struct timespec *)NULL);
        } break;
        case 2: {
            struct timespec req;
            req.tv_sec = 0;
            req.tv_nsec = 20000;
            nanosleep(&req, (struct timespec *)NULL);
        } break;
        default: ; // zero work for the others
        }
        return in;
    }
};
struct Worker2: ff_node_t<long> {
    long* svc(long*in) {
        return in;
    }
};

struct Last: ff_node_t<long> {
    long* svc(long* in) {
        printf("Last received %ld\n", (long)in);
        return GO_ON;
    }
};

int main(int argc, char* argv[]) {
    // default arguments
    size_t ntasks    = 10000;
    bool   optimize  = true;
    size_t nworkers  = 4;   // external workers
    size_t inworkers1= 2;   // internal workers of the first farm
    size_t inworkers2= 3;   // internal workers of the second farm

    if (argc>1) {
        if (argc!=6) {
            error("use: %s ntasks nworkers inworkers1 inworkers2 optimize\n",argv[0]);
            return -1;
        }
        ntasks    = std::stol(argv[1]);
        nworkers  = std::stol(argv[2]);
        inworkers1= std::stol(argv[3]);
        inworkers2= std::stol(argv[4]);
        optimize  = (std::stol(argv[5])!=0);
    }

    First first(ntasks);
    Last last;

    /* all the following farms have nworkers1 workers */
    ff_Farm<long,long> farm([&]() {	    
	    std::vector<std::unique_ptr<ff_node> > V;
	    for(size_t i=0;i<nworkers;++i) {
            // first internal farm
            std::vector<std::unique_ptr<ff_node> > W1;
            for(size_t j=0;j<inworkers1;++j)		    
                W1.push_back(make_unique<Worker1>());		
            auto ifarm1 = make_unique<ff_Farm<long,long>>(std::move(W1));

            // second internal farm
            std::vector<std::unique_ptr<ff_node> > W2;
            for(size_t j=0;j<inworkers2;++j)		    
                W2.push_back(make_unique<Worker2>());		
            auto ifarm2 = make_unique<ff_Farm<long,long>>(std::move(W2));

            auto ipipe = make_unique<ff_Pipe<long,long>>(std::move(ifarm1), std::move(ifarm2));
            V.push_back(std::move(ipipe));

	    }
	    return V;
        } ());

    // original network
    ff_Pipe<> pipe(first, farm, last);
    
    // optimization that I would like to apply, if possible
    if (optimize) {
        OptLevel opt;
        opt.max_nb_threads=ff_realNumCores();
        opt.max_mapped_threads=opt.max_nb_threads;
        opt.verbose_level=2;
        opt.no_initial_barrier=true;
        opt.no_default_mapping=true; // disable mapping if #threads > max_mapped_threads
        opt.blocking_mode     =true;   // enabling blocking if #threads > max_nb_threads
        opt.merge_farms=true;
        opt.merge_with_emitter=true;   // merging previous pipeline stage with farm emitter 
        opt.remove_collector  =true;   // remove farm collector
        opt.introduce_a2a=true;      // introduce all-2-all between two farms, if possible

        // this call tries to apply all previous optimizations modifying the pipe passed
        // as parameter
        if (optimize_static(pipe,opt)<0) {
            //if (optimize_static(farm,opt)<0) {
            error("optimize_static\n");
            return -1;
         }
    }

    printf("Pipe cardinality: %d\n", pipe.cardinality());
    
    // running the optimized pipe
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    printf("test DONE\n");
    return 0;
}
