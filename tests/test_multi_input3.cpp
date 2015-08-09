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
 *     2-stages pipeline                        
 *                                              
 *              --> Worker1 -->                                 --> Worker2-->
 *             |               |                               |              |
 *  Emitter1 -- --> Worker1 -->  --> Collector1 --> Emitter2 -- --> Worker2-->  --> Collector2                       
 *             |               |                       ^       |              |         |
 *              --> Worker1 -->                        |        --> Worker2-->          |
 *                                                     |________________________________|
 *                                    
 *                                    
 *
 */

#include <iostream>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>

using namespace ff;

class Emitter1: public ff_node {
public:
    Emitter1(int ntasks):ntasks(ntasks) {}
    void* svc(void*) {
        for(long i=0;i<ntasks;++i)
            ff_send_out((void*)(i+10));
        return NULL;
    }
private:
    int ntasks;
};

class Emitter2: public ff_node {
public:
    Emitter2(ff_loadbalancer *lb):lb(lb) {}
    void* svc(void* t) {
        int wid = lb->get_channel_id();
        if (wid == -1) {
            printf("TASK FROM INPUT %ld\n", (long)(t));
            // task coming from previous stage
            return t;
        }
        printf("got back a task from Collector\n");
        return GO_ON;
    }
    void eosnotify(ssize_t) {
        lb->broadcast_task((void*)EOS);
    }
private:
    ff_loadbalancer *lb;
};

class Worker1: public ff_node {
public:
    void* svc(void* task) {
        return task;
    }
};

class Worker2: public ff_node {
public:
    void* svc(void* task) {
        printf("Worker2(%ld): TASK COMPUTED\n", get_my_id());
        return task;
    }
};


int main(int argc, char* argv[]) {
    int nworkers = 3;
    int ntasks = 1000;
    if (argc>1) {
        if (argc < 3) {
            std::cerr << "use:\n" << " " << argv[0] << " numworkers ntasks\n";
            return -1;
        }
        nworkers  =atoi(argv[1]);
        ntasks    =atoi(argv[2]);
    }
    ff_pipeline pipe;
    ff_farm<> farm1;
    ff_farm<> farm2;
    pipe.add_stage(&farm1);
    pipe.add_stage(&farm2);
    farm1.add_emitter(new Emitter1(ntasks));
    farm1.add_collector(NULL); // default collector
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) {
        w.push_back(new Worker1);
    }
    farm1.add_workers(w);

    w.clear();
    for(int i=0;i<nworkers;++i) {
        w.push_back(new Worker2);
    }
    farm2.add_emitter(new Emitter2(farm2.getlb()));
    farm2.add_collector(NULL); // default collector
    farm2.add_workers(w);
    farm2.wrap_around(true); // true since we have multi-input in the Emitter

    pipe.run_and_wait_end();

    printf("Time= %.2f (ms)\n", pipe.ffwTime());
    return 0;
}
