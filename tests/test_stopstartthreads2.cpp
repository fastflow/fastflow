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

/**
 *
 * This is a basic test for testing the capability of starting and stopping 
 * farm's workers in the load-balancer thread.
 *
 */
#if !defined(FF_INITIAL_BARRIER)
// to run this test we need to be sure that the initial barrier is executed
#define FF_INITIAL_BARRIER
#endif
#include <vector>
#include <cstdio>
#include <ff/ff.hpp>
using namespace ff;

class Emitter: public ff_node {
protected:
    void stop_workers() {
        size_t nw = lb->getnworkers();
        for(size_t i=0; i<nw;++i)  {
            lb->ff_send_out_to(GO_OUT, i);
        }
        for(size_t i=0;i<nw;++i) {
            lb->wait_freezing(i);
        }
    }
    void wakeup_workers(bool freeze=true) {
        for(size_t i=0;i<lb->getnworkers();++i)
            lb->thaw(i,freeze);
    }
public:
    Emitter(ff_loadbalancer *const lb):lb(lb) {}
    
    int svc_init() {
        // set freezing flag to all workers
        for(size_t i=0;i<lb->getnworkers();++i)
            lb->freeze(i);
        stop_workers();

        return 0;
    }
    void *svc(void *) {
        
        for(int i=0;i<10; ++i) {
            printf("iter %d\n", i);
            usleep(i*10000);

            // restart all workers
            wakeup_workers();
            
            // do something here
            for(size_t j=0;j<lb->getnworkers();++j)
                lb->ff_send_out_to(new int(j), j);
            
            // put workers to sleep
            stop_workers();
        }
        // restart workers before sending EOS
        wakeup_workers(false);
        return EOS;
    }
    
    void svc_end() {
        printf("Emitter exiting\n");
    }
private:
    ff_loadbalancer *const lb;
};


class Worker:public ff_node {
public:

    int svc_init() {
        printf("worker %ld started\n", get_my_id());
        return 0;
    }
    void *svc(void *t) {
        printf("worker %ld received %d\n", get_my_id(), *(int*)t);
        return GO_ON;
    }
    void svc_end() {
        printf("worker %ld going to sleep\n", get_my_id());
    }
};

int main(int argc, char *argv[]) {
#if defined(BLOCKING_MODE)
    printf("TODO: mixing dynamic behavior and blocking mode has not been tested yet!!!!!\n");
    return 0;
#endif

    int nworkers = 3;
    if (argc>1) {
        if (argc!=2) {
            printf("use: %s nworkers\n",argv[0]);
            return -1;
        }
        
        nworkers = atoi(argv[1]);
    }
    ff_farm farm;
    std::vector<ff_node*> w;
    for(int i=0;i<nworkers;++i)
        w.push_back(new Worker);
    farm.add_workers(w);
    farm.add_emitter(new Emitter(farm.getlb()));
    
    farm.run();
    farm.getlb()->waitlb();

    return 0;
}
