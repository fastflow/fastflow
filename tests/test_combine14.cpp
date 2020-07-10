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


/*     A2A( pipe(farm), A2A(comb(miHelper,Collector), Sink) )  
 *
 *  |<------------------------------------ A2A ---------------------------------------->|
 *  |<--------------- pipe ----------------->|
 *  |<--------------- a2a ------------------>|   |<-------------- A2A ----------------->|
 *                 |<----- pipe ----->|            |<-------- comb ------>|
 * 
 *              | --> Worker--> Worker -->
 *  Generator-->|                         | -->|   
 *              | --> Worker--> Worker -->     | --> miHelper-->Collector --> |
 *                                             |                              |
 *                                             |                              |  --> Sink
 *              | --> Worker--> Worker -->     | --> miHelper-->Collector --> |
 *  Generator-->|                         | -->|    
 *              | --> Worker--> Worker -->
 *
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

struct miHelper: ff_minode_t<long> {
    long *svc(long*in) { return in;};   
};
struct moHelper: ff_monode_t<long> {
    long *svc(long*in) { return in;};   
};

struct Worker: ff_node_t<long> {   // NOTE: it must be a multi-output node
    long* svc(long*in) {
	printf("Worker received %ld\n", (long)in);
        return in;
    }
};
struct Collector: ff_monode_t<long> {
    Collector(int myid):myid(myid) {}
    long *svc(long *in) {
	printf("Collector%ld (myid=%d) received %ld from %ld\n", get_my_id(), myid, (long)in, get_channel_id());
	return in;
    }
    void eosnotify(ssize_t) {
	printf("Collector (myid=%d) received EOS\n", myid);
    }
    int myid=-1;
};

struct Sink: ff_minode_t<long> {
    long* svc(long*) {
	return GO_ON;
    }
};

int main(int argc, char* argv[]) {
    int nworkers=2;
    int ntasks=1000;

    if (argc!=1) {
	if (argc!=3) {
	    printf("use: %s nworkers ntasks\n", argv[0]);
	    return -1;
	}
	nworkers = atoi(argv[1]);
	ntasks    = atoi(argv[2]);
    }

    // main all-to-all
    ff_a2a a2a;

    // L-Workers ----------------
    
    ff_pipeline pipe1;
    ff_a2a* a2a1 = new ff_a2a;	
    std::vector<ff_node*> L;
    L.push_back(new Generator(ntasks));
    a2a1->add_firstset(L,0,true);
    std::vector<ff_node*> R;
    for(int i=0;i<nworkers;++i) {
        ff_pipeline* pipe = new ff_pipeline;
        pipe->add_stage(new ff_comb(new miHelper, new Worker, true, true));
        pipe->add_stage(new Worker);  // it should be multi-output, see (*)
        pipe->cleanup_nodes();
        R.push_back(pipe);
    }
    a2a1->add_secondset(R, true);
    pipe1.add_stage(a2a1);
    pipe1.cleanup_nodes();

    ff_pipeline pipe2;
    ff_a2a* a2a2 = new ff_a2a;	
    std::vector<ff_node*> L1;
    L1.push_back(new Generator(ntasks));
    a2a2->add_firstset(L1,0,true);
    std::vector<ff_node*> R1;
    for(int i=0;i<nworkers;++i) {
        ff_pipeline* pipe = new ff_pipeline;
        pipe->add_stage(new ff_comb(new miHelper, new Worker, true, true));
        pipe->add_stage(new Worker);  // it should be multi-output, see (*)
        pipe->cleanup_nodes();
        R1.push_back(pipe);
    }
    a2a2->add_secondset(R1, true);
    pipe2.add_stage(a2a2);
    pipe2.cleanup_nodes();

    std::vector<ff_node*> first_set;

    combine_with_laststage(pipe1, new moHelper, true);  // (*)
    combine_with_laststage(pipe2, new moHelper, true);  // (*)
    
    first_set.push_back(&pipe1);
    first_set.push_back(&pipe2);
    a2a.add_firstset(first_set);

    // R-Workers ----------------
    
    ff_pipeline pipe3;
    ff_a2a right_a2a;
    ff_comb* comb1 = new ff_comb(new miHelper, new Collector(1), true, true);
    ff_comb* comb2 = new ff_comb(new miHelper, new Collector(2), true, true);
    std::vector<ff_node*> rA2A_left;
    rA2A_left.push_back(comb1);
    rA2A_left.push_back(comb2);
    right_a2a.add_firstset(rA2A_left, 0 , true);
    std::vector<ff_node*> rA2A_right;
    rA2A_right.push_back(new Sink);
    right_a2a.add_secondset(rA2A_right, true);
    pipe3.add_stage(&right_a2a);
    std::vector<ff_node*> second_set;
    second_set.push_back(&pipe3);
    a2a.add_secondset(second_set);


    printf("cardinality %d\n", a2a.cardinality());
       
    if (a2a.run_and_wait_end()<0) {
        error("running a2a\n");
        return -1;
    }
    
    printf("TEST DONE\n");
    return 0;
}
