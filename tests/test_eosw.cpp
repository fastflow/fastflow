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
/* test for checking the propagation of the EOSW mark */

/* Author: Massimo Torquati
 *
 */

#include <cstdio>
#include <ff/ff.hpp>

using namespace ff;

struct First: ff_node_t<long> {   
    long *svc(long *) {
	for(size_t i=0;i<10;++i) 
	    ff_send_out((long*)(i+1));

	return EOS;
    }
    void svc_end() {
	printf("First svc_end() called\n");
    }
};

struct Last: ff_node_t<long> {
    long *svc(long *task) {
	printf("Last received %ld\n", reinterpret_cast<long>(task));
	return GO_ON;
    }
    void svc_end() {
	printf("Last svc_end() called\n");
    }
};

struct Emitter: ff_node_t<long> {
    int svc_init() {
        printf("Emitter started\n");
        return 0;
    }

    long *svc(long *task) { 
        const long &t = reinterpret_cast<long>(task);
        printf("Emitter, OS threadId=%ld  received %ld\n", getOSThreadId(), t);
        if (t == 5) return EOSW; // this is propagated to the workers
        return task; 
    }
    void svc_end() {
        printf("Emitter svc_end() called\n");
    }
};


struct Worker: ff_node_t<long> {
    long *svc(long *task) { 
	printf("Worker%ld, received %ld\n", get_my_id(), reinterpret_cast<long>(task));
	return task; 
    }
    void svc_end() {
	printf("Worker%ld, OS threadId=%ld svc_end() called\n", get_my_id(), getOSThreadId());
    }
};

struct Collector: ff_node_t<long> {
    Collector(const size_t nworkers):nworkers(nworkers),neos(0) {}
    long *svc(long *task) { 
        printf("Collector, OS threadId=%ld, received %ld\n", getOSThreadId(), reinterpret_cast<long>(task));
	return task; 
    }

    // EOSW is not propagated outside the farm so we need to explicitly send 
    // the EOS to the next stage to let him to terminate
    void eosnotify(ssize_t) {
	if (++neos >= nworkers) ff_send_out(EOS);
    }
  
    void svc_end() {
	printf("Collector svc_end() called\n");
    }

    size_t nworkers;
    size_t neos;
};


int main() {
    const size_t nworkers = 2;
    First first;
    Last  last;
    Emitter E;
    Collector C(nworkers);
    ff_Farm<long,long> farm(  []() { 
	    std::vector<std::unique_ptr<ff_node> > W;
	    for(size_t i=0;i<nworkers;++i)  W.push_back(make_unique<Worker>());
	    return W;
	} () , E, C);
    ff_Pipe<> pipe(first,farm,last);
    pipe.run_then_freeze();
    pipe.wait();

    return 0;
}
