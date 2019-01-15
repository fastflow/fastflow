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
 *  |<----------------------------- pipe ------------->|
 *  |<--------------- farm1 -------------->| 
 *                            |<-- comp -->|            
 *               |--> W -->|    Filter1+2  
 *               |         |             
 *   Emitter --> |--> W -->| --> Collector --> Gatherer
 *               |         |             
 *               |--> W -->|  
 *
 */
/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

struct Emitter: ff_monode_t<long> { 
    long *svc(long*) {
        size_t n = get_num_outchannels();
        for(long i=1;i<=10;++i) 
            ff_send_out_to((long*)i, i % n);
        return EOS;
    }
};
struct Worker: ff_node_t<long> {
    long *svc(long *in) {
        return in;
    }
};
struct Filter1: ff_node_t<long> {
    long *svc(long *in) {
        printf("Filter1: %ld\n", (long)in);
        return in;
    }
};
struct Filter2: ff_node_t<long> {
    long *svc(long *in) {
        printf("Filter2: %ld\n", (long)in);
        return in;
    }
};
struct Gatherer: ff_node_t<long> {
    long *svc(long *in) {
	printf("Gatherer: %ld\n", (long)in);
	return GO_ON;
    }
};

int main() {
    Emitter E;
    Gatherer G;
    
    std::vector<std::unique_ptr<ff_node>> W;
    W.push_back(make_unique<Worker>());
    //W.push_back(make_unique<Worker>());

    ff_Farm<> farm(std::move(W), E);
    Filter1 f1;
    Filter2 f2;
    auto comb = combine_nodes(f1, f2);
    farm.add_collector(comb);

    ff_Pipe<>  pipe(farm, G);
    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }
    return 0;
}

