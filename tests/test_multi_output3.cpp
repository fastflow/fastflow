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
 * Date:   November 2015
 */
/*
 *    
 *  Testing the following skeleton:   
 *    
 *
 *                               ---------------------
 *                               v                    |
 *                             -----                  |
 *                       ---->|     |---------        |
 *                      |     |  W  |        |        |
 *                      |      -----         v        |
 *             -----    |        .          -----     |
 *            |  E  |-->| -->    .     --> |  C  | ---|
 *             -----    |        .          -----     |
 *                      |      -----         ^        | 
 *                       ---->|  W  |        |        | 
 *                            |     |--------         |
 *                             -----                  |
 *                              ^                     |
 *                              ----------------------
 *
 *  This is a standard farm with collector plus some extra channels
 *  connected "by hand".
 *
 */      

#include <iostream>
#include <ff/multinode.hpp>
#include <ff/farm.hpp>

using namespace ff;

static const long ITER = 10;

struct Emitter: ff_node_t<long> {
    
    long* svc(long *) {
        for(long i=1;i<=ITER;++i)
            ff_send_out((void*)i);
        return EOS;
    }
    
};

struct Collector: ff_node_t<long> {
    
    Collector(ff_gatherer *gt, std::vector<ff_buffernode> &channels):
        gt(gt), channels(channels) {}
    
    long* svc(long *in) {
        long *V[2];
        
        gt->all_gather(in, reinterpret_cast<void**>(&V[0]));

        for(int i=0;i<2;++i)
            std::cout << "received from " << i << " value " << (long)V[i] << "\n";


        for(size_t i=0;i<2;++i) {
            bool r= channels[i].ff_send_out(in);
            assert(r==true);
        }
    
        return GO_ON;
    }
    
    ff_gatherer *gt;
    std::vector<ff_buffernode> &channels;
        
};

struct Worker: ff_node_t<long> {

    Worker(ff_buffernode &channel):channel(channel) {}

    long* svc(long *in) {
        std::cout << "worker" << get_my_id() << " sending " << long(in) << "\n";
        ff_send_out(in);
        long in2;
        bool r= channel.gather_task((void **)&in2);
        assert(r==true);
        std::cout << "worker" << get_my_id() << " received " << in2 << "\n";
        return GO_ON;       
    }

    void eosnotify(ssize_t id) {
        if (id != -1) return;
        ff_send_out(EOS);
    }
    ff_buffernode &channel;
};


int main() {

    std::vector<ff_buffernode> channels(2);
    channels[0].set(100,false, 0);
    channels[1].set(100,false, 1);

    std::vector<std::unique_ptr<ff_node> > W;
    W.push_back(make_unique<Worker>(channels[0]));
    W.push_back(make_unique<Worker>(channels[1]));


    ff_Farm<> farm(std::move(W));
    Emitter   E;
    Collector C(farm.getgt(), channels);
    farm.add_emitter(E);
    farm.add_collector(C);
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    
    return 0;

}
