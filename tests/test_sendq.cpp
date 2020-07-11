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
 * just a "strange" test which mixes lock and lock-free queues .....
 *
 *
 */

#include <vector>
#include <iostream>
#include <ff/platforms/platform.h>
#include <ff/ubuffer.hpp>
#include <ff/ff.hpp>
//#include <ff/staticlinkedlist.hpp>


using namespace ff;

// global lock
lock_t lock;

class Worker: public ff_node {
public:
    void * svc(void * task) {
        SWSR_Ptr_Buffer * q = (SWSR_Ptr_Buffer*)task;
        //staticlinkedlist * q = (staticlinkedlist*)task;

        bool r = false;
        do {
            /* The following lock cannot be removed. It is needed because 
             * the push method is not atomic.
             */
            spin_lock(lock);
            r = q->push(task);
            spin_unlock(lock);            
        } while(!r);
        return GO_ON;
    }
};


// the load-balancer filter
class Emitter: public ff_node {
public:
    Emitter(int max_task):q(NULL), ntask(max_task) {};
    
    int svc_init() {
        q = new SWSR_Ptr_Buffer(1024);
        if (!q->init()) return -1;
        //q = new staticlinkedlist;

        return 0;
    }
    
    void * svc(void *) {	
        void * recv_data= (void*)q;
        for(unsigned i=0;i<ntask;++i) {
            ff_send_out((void*)recv_data);
            do {} while(!q->pop(&recv_data));
            assert(recv_data == (void*)q);
        }
        return NULL;
    }
    
    void svc_end() {
        if (q) delete q;
    }
    
private:
    SWSR_Ptr_Buffer * q;
    //staticlinkedlist *q;
    unsigned ntask;
};


int main(int argc, char * argv[]) {
    int nworkers = 3;
    int streamlen=1000;
    if (argc>1) {
        if (argc<3) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nworkers ntasks\n";
            return -1;
        }        
        nworkers=atoi(argv[1]);
        streamlen=atoi(argv[2]);
    }

    if (!nworkers || !streamlen) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    init_unlocked(lock);

    ff_farm farm; // farm object
    
    Emitter E(streamlen);
    farm.add_emitter(&E);
    
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w); // add all workers to the farm
    
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    std::cerr << "DONE, time= " << farm.ffTime() << " (ms)\n";
    farm.ffStats(std::cerr);
    
    return 0;
}
