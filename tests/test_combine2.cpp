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
 *
 *
 */

/*  pipe(farm(comb1), comb2)
 *
 *
 *  |<------ farm without collector ------->|   |<--------- comb2 --------->|
 *  |                |<---- comb1 ---->|        |<- multi-input ->|
 *                   
 * 
 *               |--> Worker1-->Worker2 -->|  
 *               |                         |  
 *   Emitter --> |--> Worker1-->Worker2 -->|-------> Filter1 ---------> Filter2
 *               |                         |  
 *               |--> Worker1-->Worker2 -->|  
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
        for(long i=1;i<=100;++i) 
            ff_send_out_to((long*)i, i % n);
        return EOS;
    }
};
struct Worker1: ff_node_t<long> {
    long *svc(long *in) {
        return in;
    }
};
struct Worker2: ff_node_t<long> {
    long *svc(long *in) {
        printf("Worker2 (id=%ld) in=%ld\n", get_my_id(), (long)in);
        return in;
    }
};
struct Filter1: ff_minode_t<long> {  
    long *svc(long *in) {
        std::cout << "received input from " << get_channel_id() << "\n";
        return in;
    }
};

struct Filter2: ff_node_t<long> {
	long *svc(long *in) {
        std::cout << get_my_id() << ": " << (long)in << "\n";
        return GO_ON;
    }
};

int main() {

    Emitter E;

    Worker1 w11, w12, w13;
    Worker2 w21, w22, w23;
    std::vector<std::unique_ptr<ff_node>> W;
    
    W.push_back(unique_combine_nodes(w11,w21));
    W.push_back(unique_combine_nodes(w12,w22));
    W.push_back(unique_combine_nodes(w13,w23));

    
    ff_Farm<> farm(std::move(W), E);
    farm.remove_collector();

    Filter1 f1;
    Filter2 f2;
    
    ff_Pipe<> pipe(farm, combine_nodes(f1,f2));

    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }
    
    return 0;
}

