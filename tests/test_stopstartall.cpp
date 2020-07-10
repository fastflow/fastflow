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
 * Date:   February 2014
 */
/**
 * This test shows how to stop all farm threads and then restart a subset
 * of worker threads. At the beginning the farm is created with the
 * maximum number of workers (i.e. ff_numCore()) and immediatly all threads
 * are frozen, then a different number of worker threads will be restarted
 * at each iteration of an external loop.
 *
 */
#include <vector>
#include <cstdio>
#include <ff/mapping_utils.hpp>
#include <ff/ff.hpp>

using namespace ff;


static volatile bool reconfGo;
// an "external" callback reading the reconfGo flag
static inline bool checkReconfCondition() {
    return reconfGo;   
}


struct Emitter: ff_node {
    Emitter(long ntasks):ntasks(ntasks) {}
    int svc_init() {
        printf("Emitter: woken-up\n");
        return 0;
    }
    void *svc(void *) {
        if (checkReconfCondition()) {
            printf("Emitter: RECONF STARTED\n");
            return NULL;
        }
        for(long i=1;i<=ntasks;++i) 
            ff_send_out((void*)i);
        usleep(5000);
        return GO_ON;
    }
    void svc_end() {
        printf("Emitter: going to sleep\n");
    }

    long ntasks;
};
struct Collector: ff_node {
    int svc_init() {
        printf("Collector: woken-up\n");
        ntasks=0;
        return 0;
    }
    void *svc(void *) {
        ++ntasks;
        return GO_ON;
    }
    void svc_end() {
        printf("Collector: going to sleep, received %ld tasks\n",ntasks);
    }
    long ntasks;
};
struct  Worker: ff_node {
    int svc_init() {
        printf("Worker%ld: woken-up\n",get_my_id());
        return 0;
    }
    void *svc(void *t) {
        //printf("Worker%ld: received %ld\n", get_my_id(), (long)t);
        return t;
    }
    void svc_end() {
        printf("Worker%ld: going to sleep\n", get_my_id());
    }
};

int main(int argc, char *argv[]) {
#if defined(BLOCKING_MODE)
    printf("TODO: mixing dynamic behavior and blocking mode has not been tested yet!!!!!\n");
    return 0;
#endif

    int maxworkers   = ff_numCores();
    int ntasks       = 100;
    int nstopstart   = 10;
    if (argc>1) {
        if (argc!=4) {
            printf("use: %s maxworkers ntasks nstopstart\n",argv[0]);
            return -1;
        }
        
        maxworkers = atoi(argv[1]);
        ntasks     = atoi(argv[2]);
        nstopstart = atoi(argv[3]);
    }

    ff_farm farm;
    std::vector<ff_node*> w;
    for(int i=0;i<maxworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);
    farm.add_emitter(new Emitter(ntasks));
    farm.add_collector(new Collector);

    // warm-up: all farm threads start and then stop
    reconfGo = true;
    farm.run_then_freeze();
    farm.wait_freezing();
    printf("------warm-up: done\n\n");

    bool up = true;
    int n = 1; // starts with 1 worker out of maxworkers
    reconfGo = false;
    for(int i=0;i<nstopstart;++i) {
        printf("running with n=%d workers\n", n);
        farm.run_then_freeze(n);
        usleep(300000); // waits a while before starting reconfiguration
        reconfGo = true;
        farm.wait_freezing();
        if (up && n<maxworkers)
            n = (std::min)(2*n, maxworkers);
        else {
            n = (std::max)(1, n/2);
            up=false; // starting to go down
        }

        reconfGo = false; // resets the flag
    }

    farm.wait();
    return 0;
}
