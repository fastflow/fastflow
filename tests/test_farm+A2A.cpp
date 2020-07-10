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
/* This is an example showing how to combine the farm and the all-to-all building 
 * blocks for implementing an interesting pattern. It looks like as a logical farm having 
 * pipeline chains as workers. However, in this case the farm has a master-worker configuration 
 * so the scheduler may implement an on-demand policy (or whatever else policy) while 
 * the workers are pipelines implemented by using the L-Worker set of the all-to-all building block. 
 * The collector of the logical farm is implemented by a single R-Worker of the all-to-all.
 *
 *      pipe( feedback(farm1(Worker)), A2A(pipeline(First,Second), Collector) )
 *
 * 
 *        ___________                    
         |           |
 *       |      |--> Worker --->  First --> Second -->|
 *       v      |                                     |
 *  Scheduler-->|--> Worker --->  First --> Second -->| ---> Collector
 *     ^ ^      |          |                          |
 *     | |      |--> Worker --->  First --> Second -->|
 *     | |___________|     |    
 *     |___________________|     |<---- pipeline --->|  |<- R-Worker ->|  
 *                                  (L-Workers)
 *                       
 *  |<- farm master-worker -->| |<------------- all-to-all ----------->|                          
 *        (no collector)
 *
 *
 */
/*
 * Author: Massimo Torquati
 *
 */

#include <ff/ff.hpp>

using namespace ff;

// this is a simple scheduling policy, for a more complex implementation
// take a look at test_multi_output5.cpp test. 
struct Scheduler: ff_minode_t<long> {
    long* svc(long* in) {
	if (in == nullptr) {
	    for(long i=1;i<=ntasks;++i)
		ff_send_out((long*)i);
	    return GO_ON;
	}
	if (++cnt>=ntasks) return EOS;
	return GO_ON;
    }
    long cnt=0;
    const long ntasks= 1000;
}; 
struct Worker: ff_monode_t<long> {
    long* svc(long* in) {
	ff_send_out_to(in, 0); // back
	ff_send_out_to(in, 1); // forward
	return GO_ON;
    }
};
struct First: ff_node_t<long> {
    long* svc(long* in) { return in;}
};
// The last stage of the pipeline must be multi-output
// if it is used as L-Worker.
struct Second: ff_monode_t<long> {
    long* svc(long* in) { return in;}
};
struct Collector: ff_minode_t<long> {
    long* svc(long*) {
	printf("Collector, received from pipe%ld\n", get_channel_id());
	return GO_ON;
    }
};

int main() {
    const int nworkers=3;
    // preparing the master-worker farm
    Scheduler sched;
    std::vector<ff_node*> W;
    for(int i=0;i<nworkers;++i) 
        W.push_back(new Worker);
    ff_farm farm(W,&sched);
    farm.remove_collector();
    farm.cleanup_workers();
    farm.wrap_around();
    
    // preparing the all-to-all
    ff_a2a a2a;
    std::vector<ff_node*> W1;
    for(int i=0;i<nworkers;++i) {
        ff_pipeline *pipe = new ff_pipeline;
        pipe->add_stage(new First,  true);
        pipe->add_stage(new Second, true);
        W1.push_back(pipe);
    }
    a2a.add_firstset(W1, 0, true);
    Collector col;
    std::vector<ff_node*> W2;
    W2.push_back(&col);
    a2a.add_secondset(W2);
    

    ff_Pipe<> pipe(farm, a2a);
    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }
    return 0;
}
