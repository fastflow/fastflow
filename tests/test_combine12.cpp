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
 *
 */

#include <ff/ff.hpp>

using namespace ff;

const long WORKTIME_TICKS=25000;

struct Generator: ff_node_t<long> {
    Generator(long ntasks):ntasks(ntasks) {}
    long *svc(long *) {
	for(long i=1;i<=ntasks;++i)
	    ff_send_out((long*)i);
        return EOS;
    }
    long ntasks;
};

struct Worker1: ff_node_t<long> {
    long* svc(long*in) {
	printf("Worker1 received %ld\n", (long)in);
        return in;
    }
};
struct Worker2: ff_node_t<long> {
    Worker2(const size_t nworkers):nworkers(nworkers) {}
    long* svc(long*in) {
	if (((long)in % nworkers) != (size_t)get_my_id()) {
	    error("WRONG INPUT FOR WORKER%ld, received=%ld\n", get_my_id(), (long)in);
	    abort();
	}
	ticks_wait(WORKTIME_TICKS); 		
        return in;
    }
	const size_t nworkers;
};

struct Emitter: ff_monode_t<long> {
    long *svc(long *in) {
	printf("Emitter received %ld\n", (long)in);
        ff_send_out_to(in, (long)in % get_num_outchannels());
        return GO_ON;
    }
};
struct Collector1: ff_node_t<long> {
    int svc_init() {
	return 0;
    }
    long *svc(long *in) {
	printf("Collector received %ld\n", (long)in);
	if ((long)in != cnt++) abort();

        return in;
    }
    long cnt=1;
};
struct Collector2: ff_node_t<long> {
    long *svc(long *) {
        return GO_ON;
    }
};


int main(int argc, char* argv[]) {
    int nworkers1=2;
    int nworkers2=3;
    int ntasks=1000;

    if (argc!=1) {
	if (argc!=4) {
	    printf("use: %s nworkers1 nworkers2 ntasks\n", argv[0]);
	    return -1;
	}
	nworkers1 = atoi(argv[1]);
	nworkers2 = atoi(argv[2]);
	ntasks    = atoi(argv[3]);
    }
    
    // first farm, it's an ordered farm
    ff_farm* farm1 = new ff_farm;	
    farm1->add_emitter(new Generator(ntasks));
    std::vector<ff_node*> W1;
    for(int i=0;i<nworkers1;++i)
	W1.push_back(new Worker1);
    farm1->add_workers(W1);
    farm1->add_collector(new Collector1); 
    farm1->cleanup_all();
    farm1->set_ordered();
    
    // second farm
    ff_farm* farm2 = new ff_farm;	
    farm2->add_emitter(new Emitter);    
    farm2->add_collector(new Collector2);
    std::vector<ff_node*> W2;
    for(int i=0;i<nworkers2;++i)
	W2.push_back(new Worker2(nworkers2));
    farm2->add_workers(W2);
    farm2->cleanup_all();

    auto pipe = combine_ofarm_farm(*farm1, *farm2);
    if (pipe.run_and_wait_end()) {
	error("running pipeline\n");
	return -1;
    }
    delete farm1;
    delete farm2;
    
    printf("TEST DONE\n");
    return 0;
}
