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
/* Author: Massimo Torquati
 *         torquati@di.unipi.it
 *
 */

// This simple test shows how to re-use the same task-farm engine for different computations
// (in this case kernel1 and kernel2) by using the ff_nodeSelector node.
// Another way is to uses lambdas as in the ff_taskf pattern.
//

#include <ff/ff.hpp>
#include <ff/selector.hpp>  

// first kernel 
struct kernel1: ff::ff_node_t<long> {
    long *svc(long *in) {
	printf("kernel1 %ld\n", *in);
	return GO_ON;
    }
};

// second kernel
struct kernel2: ff::ff_node_t<long> {
    long *svc(long *in) {
	printf("kernel2 %ld\n", *in);
	return GO_ON;
    }
};

// the task-farm uses the same Emitter (it is possible to use multiple Emitter as well)
struct Emitter: ff::ff_node_t<long> {
    long *svc(long *) {
	for(long i=0;i<20;++i)
	    ff_send_out(new long(i));
	return EOS;
    }
};

int main() {
	using namespace ff;

    const size_t farmworkers = 1;
    Emitter E;

    // create the farm (3 Workers)
    ff_Farm<> farm([]() {
	    std::vector<std::unique_ptr<ff_node> > W;
	    for(size_t i=0;i<farmworkers;++i)
	    	W.push_back(ff::make_unique<ff_nodeSelector<long>>(ff::make_unique<kernel1>(), ff::make_unique<kernel2>()));
	    return W;
	} (), E);
    farm.remove_collector();

    // select first the kernel2 (1) and then the kernel1 (0)

    const svector<ff_node*>& W = farm.getWorkers();
    for(size_t i=0;i<W.size();++i)
	(reinterpret_cast<ff_nodeSelector<long> *>(W[i]))->selectNode(1);

    printf("/* ----------------------- kernel2 start --------------------- */\n");
    farm.run_then_freeze();
    farm.wait_freezing();
    printf("/* ----------------------- kernel2 done ---------------------- */\n");

    for(size_t i=0;i<W.size();++i)
	(reinterpret_cast<ff_nodeSelector<long> *>(W[i]))->selectNode(0);

    printf("/* ----------------------- kernel1 start --------------------- */\n");
    farm.run_then_freeze();
    farm.wait_freezing();
    printf("/* ----------------------- kernel1 done ---------------------- */\n");

    farm.wait();
    return 0;
}
