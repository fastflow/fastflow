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
 * 
 *                               |-->Worker1 --->|
 *                               |               |
 *              |--> Emitter1 -->|-->Worker1 --->|
 *              |                                |
 *  Scheduler-->|                                |
 *     ^        |--> Emitter2--->|-->Worker2 --->|
 *     |                         |               |
 *     |                         |-->Worker2 --->|
 *     |_________________________________________|
 *                       
 *
 *
 */
/*
 * Author: Massimo Torquati
 *
 */

#include <ff/ff.hpp>

using namespace ff;

// this is a simple scheduling policy, for a more complex implementation
// take a look at test_multi_output5.cpp test. 
struct Scheduler: ff_minode_t<long> {
    long* svc(long* in) {
        if (in == nullptr) {
            for(long i=1;i<=ntasks;++i)
                ff_send_out((long*)i);
            return GO_ON;
        }
        printf("Scheduler got back %ld from %ld\n", (long)in, get_channel_id()); 
        if (++cnt>=ntasks) return EOS;
        return GO_ON;
    }
    long cnt=0;
    const long ntasks= 1000;
}; 
struct Worker: ff_node_t<long> {
    long* svc(long* in) {
        printf("Worker%ld sends back %ld\n", get_my_id(), (long)in);
        return in;
    }
};
struct Emitter: ff_node_t<long> {
    long* svc(long* in) { 
        return in;
    }
};

int main() {
    ff_farm farm1, farm2;
    Emitter E1, E2;
    farm1.add_emitter(&E1);
    farm2.add_emitter(&E2);
    std::vector<ff_node*> W1;
    W1.push_back(new Worker);
    //W1.push_back(new Worker);
    farm1.add_workers(W1);
    W1.clear();
    W1.push_back(new Worker);
    //W1.push_back(new Worker);
    farm2.add_workers(W1);


    ff_farm farm;
    Scheduler sched;
    farm.add_emitter(&sched);
    std::vector<ff_node*> W;
    W.push_back(&farm1);
    W.push_back(&farm2);
    farm.add_workers(W);
    farm.wrap_around();
        
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    return 0;
}
