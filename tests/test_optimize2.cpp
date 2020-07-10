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
 *  pipe(First, Farm(Farm), Last)   
 *
 *  All farms collectors are removed and the last stage transformed to multi-input node, 
 *  the first stage is merged with the farm emitter.
 *
 *
 *                           | -> Worker -> |
 *            | defEmitter ->|              |
 *            |              | -> Worker -> |
 *            |                             |
 *            |              | -> Worker -> |
 *   First -->| defEmitter ->|              | --> Last
 *            |              | -> Worker -> |
 *            |                             |
 *            |              | -> Worker -> |
 *            | defEmitter ->|              |
 *                           | -> Worker -> |
 *
 */
/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

struct First: ff_node_t<long> {
    long* svc(long*) {
	for(long i=1;i<=1000;++i)
	    ff_send_out((long*)i);
	return EOS;
    }
};
struct Worker: ff_node_t<long> {
    long* svc(long*in) {
        if (get_my_id()==0 || get_my_id() == 1) usleep(1000);
        printf("Worker%ld received %ld\n", get_my_id(), (long)in);
        return in;
    }
};
struct Last: ff_node_t<long> { 
    long* svc(long*in) {
        //printf("Last received %ld from %ld\n", (long)in, get_channel_id());  // <<< to print the input channel you need ff_minode-t
        printf("Last received %ld\n", (long)in);
        return GO_ON;
    }
};


int main() {
    First first;
    Last last;
    size_t nworkers1 = 3;  // n. of workers of the first 4 farms
    size_t nworkers2 = 2;  // n. of workers of farm5 and farm6


    ff_Farm<long,long> farm([&]() {
            std::vector<std::unique_ptr<ff_node> > V;
            for(size_t i=0;i<nworkers1;++i) {
                std::vector<std::unique_ptr<ff_node> > W;
                for(size_t j=0;j<nworkers2;++j) {
                    W.push_back(make_unique<Worker>());
                }
                auto F = make_unique<ff_Farm<long,long> >(std::move(W));
                V.push_back(std::move(F));
            }
            return V;
        } ());
    
    // original network
    ff_Pipe<> pipe(first, farm, last);

    // optimization that I would like to apply, if possible
    OptLevel opt;
    opt.verbose_level=1;
    opt.max_nb_threads=ff_realNumCores();
    opt.blocking_mode=true;      // enabling blocking if #threads > max_nb_threads
    opt.merge_with_emitter=true; // merging previous pipeline stage with farm emitter 
    opt.remove_collector=true;   // remove farm collector

    // this call tries to apply all previous optimizations modifying the pipe passed
    // as parameter 
    if (optimize_static(pipe,opt)<0) {
        error("optimize_static\n");
        return -1;
    }
    // running the optimized pipe
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    printf("test DONE\n");
    return 0;
}
