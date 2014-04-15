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
/* Author: Massimo
 * Date  : April 2014
 *
 */                                    
/*
 *                             
 *        -----                                -----     
 *       |     |                              |     |    
 *       |  W1 | ----                     ----|  W2 | ---
 *        -----      |                   |     -----     |
 *          .        |       -----       |       .       |      -----
 *          .        |----> |     |      |       .       |     |     |
 *          .        |      |  E  | -----|       .       |---->|  C  |
 *        -----      |       -----       |     -----     |      -----
 *       |     |-----          ^          ----|     |----         |
 *       |  W1 |               |              |  W2 |             |
 *        -----                |               -----              |
 *                              ----------------------------------
 *
 *   2-stage pipeline: 
 *     - the first stage is a farm with no Emitter and Collector
 *     - the second stage is a farm with the Collector and feedback channel
 *     - the Collector C uses the gather-all
 *
 */

#include <ff/pipeline.hpp>
#include <ff/farm.hpp> 

using namespace ff;
long   NUMTASKS=10;
int    FARM1WORKERS=2; 


struct W1: ff_node {
    void *svc(void*) {
        for(long i=(get_my_id()+1); i<=NUMTASKS; ++i) {
            ff_send_out((void*)i);
        }
        return NULL;
    }
};

struct W2: ff_node {
    void *svc(void *task) {
        long t = (long)task;
        assert(t>1);
        --t;
        return (void*)t;
    }
};

class E: public ff_node {
public:
    E(ff_loadbalancer *const lb):neos(0),numtasks(0),lb(lb) {}

    void *svc(void *task) {
        if (lb->get_channel_id() == -1) {
            long t = (long)task;
            if (t == 1) return GO_ON;            
            ++numtasks;
            printf("INPUT: sending %ld to worker\n", t);
            lb->broadcast_task((void*)t);
            return GO_ON;
        }
        long t = (long)(((long**)task)[0]);
        printf("BACK: got  %ld from collector (numtasks=%ld)\n", t,numtasks);
        
        if ((t != 1) && (t & 0x1)) {
            lb->broadcast_task((void*)t);
            return GO_ON;
        }
        --numtasks;
        if (numtasks == 0) return NULL;
        return GO_ON;
    }

protected:
    int neos;
    long numtasks;
    ff_loadbalancer *const lb;
};

class C: public ff_node {
public:
    C(ff_gatherer *const gt):gt(gt) {}

    void *svc(void *task) {
        long **V = new long*[FARM1WORKERS];
        
        gt->all_gather(task, (void**)&V[0]);
        return V;
    }

protected:
    ff_gatherer *const gt;
};


int main(int argc, char *argv[]) {

    if (argc > 1) {
        NUMTASKS=atol(argv[1]);
    }

    ff_farm<> farm1;
    std::vector<ff_node*> w;
    for(int i=0;i<FARM1WORKERS;++i)
        w.push_back(new W1);
    farm1.add_workers(w);
    farm1.remove_collector();

    ff_farm<> farm2;
    w.clear();
    w.push_back(new W2);
    w.push_back(new W2);
    farm2.add_workers(w);
    farm2.add_emitter(new E(farm2.getlb()));
    farm2.add_collector(new C(farm2.getgt()));
    farm2.setMultiInput();
    farm2.wrap_around(true);

    ff_pipeline pipe;
    pipe.add_stage(&farm1);
    pipe.add_stage(&farm2);

    pipe.run_and_wait_end();

    printf("DONE\n");
    return 0;
}
