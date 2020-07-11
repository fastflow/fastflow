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
/* Testing: 
 *   A2A( A2A(pipe(combine), pipe(combine)), pipe(combine))
 * 
 *   Source0-> Source1 -->|    |->mi-> Worker1->mo ->|
 *                        |    |         ....        |
 *        ...             | -->|                     |---> mi->Sink
 *                        |    |->mi-> WorkerM->mo ->|
 *   Source0-> Source1 -->|
 *                          
 *  |<--- pipe(comb) --->|       |<-  pipe(comb) ->|   |<-pipe(comb)->|
 *  |<----------------- pipe(all-to-all) ------------>|
 *  |<--------------- all-to-all ----------------------------------->|
 */
/*
 *
 * Author: Massimo Torquati
 */

#include <mutex>
#include <iostream>
#include <string>
#include <ff/ff.hpp>

using namespace ff;

// only for printing
std::mutex mtx;

static inline unsigned long current_time_nsecs() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}
static void active_delay_ns(unsigned long waste_time) {
    auto start_time = current_time_nsecs();
    bool end = false;
    while (!end) {
	auto end_time = current_time_nsecs();
	end = (end_time - start_time) >= waste_time;
    }
}
// -------------------------------------------

// multi-input helper node
struct miHelper:ff_minode_t<long> {
    long* svc(long* in) {
	return in;
    }
};
// multi-output helper node
struct moHelper:ff_monode_t<long> {
    long* svc(long* in) {
	return in;
    }    
};
// source pipeline
struct SourcePipe: ff_pipeline {
    struct Source0: ff_monode_t<long> {   
	Source0(const long n, std::function<void (long&)> F):ntasks(n),F(F) {}
	
	int svc_init() {
	    std::lock_guard<std::mutex> lck (mtx);
	    std::cout << "Source" << get_my_id() << " running on core " << ff_getMyCore() << "\n";
	    if (get_my_id() == 0) ff::ffTime(ff::START_TIME);
	    return 0;
	}	
	long* svc(long*) {
	    if (ntasks<=0) return EOS;	
	    long *x = new long;
	    F(*x);
	    --ntasks;
        return x;
	}
	long ntasks;	
	std::function<void (long&)> F;    
    };
    struct Source1: ff_monode_t<long> {
	int svc_init() {
	    nout = get_num_outchannels();
	    return 0;
	}
	long* svc(long* in) {
	    ff_send_out_to(in, cnt++ % nout);
	    return GO_ON;
	}
	long nout,cnt = 0;
    };
    SourcePipe(const long n, std::function<void (long&)> F) {
	add_stage(new ff_comb(new Source0(n,F), new Source1, true, true));
    }
};
// worker properly combined with helper nodes
struct WorkerPipe: ff_pipeline {
    struct Worker:ff_node_t<long> {
	int svc_init() {
	    std::lock_guard<std::mutex> lck (mtx);
	    std::cout << "Worker" << get_my_id() << " running on core " << ff_getMyCore() << "\n";
	    return 0;
	}
        long* svc(long* in) {
	    active_delay_ns(3000);
            return in;
        }
    };
    WorkerPipe() {
	add_stage(new ff_comb(new ff_comb(new miHelper, new Worker, true, true), new moHelper), true);
    }
};

struct SinkPipe: ff_pipeline {    
    struct Sink: ff_minode_t<long> {
	int svc_init() {
	    std::lock_guard<std::mutex> lck (mtx);
	    std::cout << "Sink running on core " << ff_getMyCore() << "\n";
	    return 0;
	}    
	long* svc(long* in) {
	    ++cnt;
	    delete in;	
	    return GO_ON;
	}
	void svc_end() {
	    ff::ffTime(ff::STOP_TIME);
	    std::cout << "Sink ntask= " << cnt << " Time= " << ff::ffTime(ff::GET_TIME) << " (ms)\n";	
	}
	long cnt=0;
    };
    SinkPipe() {
	add_stage(new ff_comb(new miHelper, new Sink, true, true), true);
    }
};

int main(int argc, char* argv[]) {
    long ntasks   = 1000000;
    long nsources = 4;
    long nworkers = 3;
    std::string thread_mapping="";

    if (argc>1) {
	if (argc<4) {
	    std::cerr << "Use: " << argv[0]
		      << " ntasks nsources nworkers [thread_mapping_string]\n";
	    return -1;
	}
	ntasks   = std::stol(argv[1]);
	nsources = std::stol(argv[2]);
	nworkers = std::stol(argv[3]);
	
	if (argc == 5) thread_mapping = std::string(argv[4]);
    }
    
    ff_a2a a2a_internal;
    std::vector<ff_node*> W1;
    for(long i=0;i<nsources;++i) {
	W1.push_back(new SourcePipe(ntasks/nsources, [](long&x){x=13;} ));
    }
    a2a_internal.add_firstset(W1,0,true);
    std::vector<ff_node*> W2;
    for(long i=0;i<nworkers;++i)
	W2.push_back(new WorkerPipe);
    a2a_internal.add_secondset(W2,true);

    ff_pipeline pipe_internal;
    pipe_internal.add_stage(&a2a_internal);
  
    ff_a2a a2a_main;
    W1.clear();
    W2.clear();
    W1.push_back(&pipe_internal);
    W2.push_back(new SinkPipe);
    a2a_main.add_firstset(W1);
    a2a_main.add_secondset(W2, true);
    
    if (thread_mapping != "")
	threadMapper::instance()->setMappingList(thread_mapping.c_str());
    
    if (a2a_main.run_and_wait_end()<0) abort();

    return 0;
}



