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
 * This test checks the remove_collector optimization.
 *
 *   pipe(Gen, farm( pipeline(farm1(seq), farm2(A2A(seq, pipeline(farm))))), Col )
 *
 *
 *
 */
/* Author: Massimo Torquati
 *
 */

#include <ff/ff.hpp>
using namespace ff;


struct Generator: ff_node_t<long> {
    long* svc(long*) {
        for(long i=1;i<=10;++i) {
            ff_send_out((long*)i);
        }
        return EOS;
    }
};

struct Collector: ff_minode_t<long> {
    int svc_init() {
	return 0;
    }
    long* svc(long*) {
	return GO_ON;
    }
};
struct Worker: ff_node_t<long> {
    long* svc(long*) {
	return GO_ON;
    }
};

struct PipeWorker: ff_pipeline {
    PipeWorker() {
	ff_farm* farm1 = new ff_farm;
	std::vector<ff_node*> W;
	W.push_back(new Worker);
	W.push_back(new Worker);
	farm1->add_workers(W);
	farm1->add_collector(nullptr); 
	
	ff_a2a* a2a = new ff_a2a;
	std::vector<ff_node*> Wa2a1;
	Wa2a1.push_back(new Worker);
	Wa2a1.push_back(new Worker);
	a2a->add_firstset(Wa2a1);

	
      	ff_pipeline *ipipe = new ff_pipeline;
	
	ff_farm* farm2 = new ff_farm;
	std::vector<ff_node*> W2;
	W2.push_back(new Worker);
	W2.push_back(new Worker);
	farm2->add_workers(W2);
	farm2->add_collector(nullptr);

	ipipe->add_stage(farm2);

	std::vector<ff_node*> Wa2a2;
	Wa2a2.push_back(ipipe);
	a2a->add_secondset(Wa2a2);

       
	ff_farm* farmA2A = new ff_farm;
	farmA2A->add_collector(nullptr);
	std::vector<ff_node*> _W;
	_W.push_back(a2a);
	farmA2A->add_workers(_W);

	add_stage(farm1);
	add_stage(farmA2A);
    }


    long* svc(long*) { abort();}
};


int main() {
    Generator gen;
    Collector col;

    ff_farm farm;
    std::vector<ff_node*> W;
    W.push_back(new PipeWorker);
    W.push_back(new PipeWorker);
    W.push_back(new PipeWorker);
    farm.add_workers(W);
    farm.add_collector(nullptr);

    ff_Pipe<> pipe(gen, farm, col);
    printf("Number of nodes= %d\n", pipe.cardinality());
    OptLevel opt;
    opt.remove_collector = true;
    opt.verbose_level=2;
    if (optimize_static(pipe, opt)<0) {
	error("optimizing pipe\n");
	return -1;
    }

    printf("Number of nodes= %d\n", pipe.cardinality());
    if (pipe.cardinality() != 30) {
	printf("WRONG RESULT\n");
	return -1;
    }
    if (pipe.run_and_wait_end()<0) {
	error("running pipe\n");
	return -1;
    }
    return 0;
}
