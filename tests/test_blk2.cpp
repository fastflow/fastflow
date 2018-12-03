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
/* testing BLK and NBLK */

/* Author: Massimo Torquati
 *  loopback(farm(Scheduler, Worker)) no collector
 */

#include <cstdio>
#include <ff/ff.hpp>

using namespace ff;

struct Scheduler: ff_node_t<long> {   
    Scheduler(ff_loadbalancer *lb):lb(lb) {}
    long *svc(long *in) {
        static size_t rounds = 0;
        
        if (in == nullptr) {
            // enforces nonblocking mode since the beginning
            // regardless the compilation setting
            lb->broadcast_task(NBLK);

            for(size_t i=0;i<1000;++i) 
                ff_send_out((long*)(i+1));
            ++rounds;
            return BLK;
        }

        for(size_t i=rounds*1000;i<(rounds*1000+1000);++i) 
            ff_send_out((long*)(i+1));
        ++rounds;
        switch(rounds) {
        case 2: return NBLK;
        case 3: return BLK;
        case 4: return NBLK;
        };
        return EOS;
    }
    ff_loadbalancer *const lb;
};

struct Worker: ff_node_t<long> {
    long *svc(long *task) { 
        printf("Worker%ld, received %ld\n", get_my_id(), reinterpret_cast<long>(task));
        usleep(get_my_id()*50000);
        return task; 
    }
};


int main() {
    const size_t nworkers = 3;
    ff_Farm<> farm(  [nworkers]() { 
	    std::vector<std::unique_ptr<ff_node> > W;
	    for(size_t i=0;i<nworkers;++i)  W.push_back(make_unique<Worker>());
	    return W;
	} () );
    Scheduler sched(farm.getlb());
    farm.add_emitter(sched);
    farm.remove_collector();
    farm.setFixedSize(true);
    farm.setInputQueueLength(1);
    farm.setOutputQueueLength(1);
    farm.wrap_around();
    farm.run_and_wait_end();
    return 0;
}
