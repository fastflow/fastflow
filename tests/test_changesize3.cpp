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
 *
 *
 */

/*  farm( First, feedback(A2A), Last )
 *                 
 *            ____________________________________________
 *           |                                            |
 *           |     -----------------------------------    |
 *           |    |                                   |   |
 *           |    |  |-> Worker1  ->|                 |   |
 *            --->|  |              | -> Worker2 -->| |--- 
 *                |  |-> Worker1  ->|               | |
 *     First ---->|  |              | -> Worker2 -->| |-----> Last
 *                |  |-> Worker1  ->|               | |
 *                |  |              | -> Worker2 -->| |
 *                |  |-> Worker1  ->|                 |
 *                |                                   |
 *                 -----------------------------------
 *
 */

/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

// NOTE: (NTASKS mod NWORKER1) must be 0!
const int NTASKS = 400;
const int NWORKER1 = 4;
const int NWORKER2 = 3;

std::atomic<long> feedbackcounter{0};
std::atomic<bool> stopsendingback{false};

struct First: ff_monode_t<long> {
    int svc_init() {
        svector<ff_node*> w;
        this->get_out_nodes(w);
        size_t oldsz;
        for(size_t i=0;i<w.size(); ++i)
            w[i]->change_inputqueuesize(1, oldsz);
        return 0;
    }

    long* svc(long*) {
        for(int k=0;k<NTASKS;++k) {
            ff_send_out_to(new long(k), k % NWORKER1);
        }
        return EOS;
    }
};


struct MultiInputHelper1: ff_minode_t<long> {
	long *svc(long *in) {
        return in;
    }

};

struct Worker1: ff_monode_t<long> {
    int svc_init() {
        svector<ff_node*> w;
        this->get_out_nodes(w);
        size_t oldsz;
        for(size_t i=0;i<w.size(); ++i)
            w[i]->change_inputqueuesize(1, oldsz);
        return 0;
    }
    long *svc(long *in) {
        if (get_my_id() == 0) usleep(5000);
        ff_send_out(in);
        return GO_ON;
    }
};

struct MultiInputHelper2: ff_minode_t<long> {
	long *svc(long *in) { return in; }
};

struct Worker2: ff_monode_t<long> {
    int svc_init() {
        svector<ff_node*> w;
        this->get_out_nodes(w);

        assert(w.size()==1);

        size_t oldsz, _osz = w[0]->get_in_buffer()->buffersize();
        w[0]->change_inputqueuesize(1, oldsz);
        std::printf("Worker2 %ld init, oldsz=%ld, (real old size=%lu), new size=%ld\n", get_my_id(), oldsz, _osz, w[0]->get_in_buffer()->buffersize());
        return 0;        
    }
    void sendEOS() {
        for(int i=0;i<NWORKER1; ++i)
            ff_send_out(EOS, i);
    }

    long* svc(long* in) {
        if (stopsendingback.load()) {
            if (!eossent) {
                sendEOS();
                eossent=true;
            }

            return GO_ON;
        }
        if (get_my_id() == 0) usleep(5000);

        ff_send_out(in);  // sends the task back
        if (feedbackcounter.fetch_add(1, std::memory_order_relaxed) == (NTASKS-1)) { // last one
            sendEOS();
            eossent=true;
            stopsendingback.store(true);
        }         
        return GO_ON;
    }
    
    void eosnotify(ssize_t) {
        std::printf("Worker2 id = %ld, received EOS\n", get_my_id());
        
        ff_send_out_to(new long(get_my_id()), NWORKER1);  // forward channel
    }
    bool eossent=false;
};

struct Last: ff_minode_t<long> {
    long* svc(long* in) {
        std::printf("Last received something %ld\n", *in);
        ++cnt;
        delete in;
        return GO_ON;
    }
    void eosnotify(ssize_t id) {
        std::printf("Last received EOS from %ld\n", id);
    }
    
    void svc_end() {
        if (cnt != NWORKER2) {
            std::cerr << "Error received " << cnt << " instead of " << NWORKER2 << "\n";
            exit(-1);
        }
    }
    
    long cnt=0;
};

int main() {
    // ---- first stage (Emitter)
    First first;

    // ----- building all-to-all

    const MultiInputHelper1 helper1;
    const MultiInputHelper2 helper2;
    const Worker1 w1;
    auto comb1 = combine_nodes(helper1, w1);
    auto comb2 = combine_nodes(helper1, w1);
    auto comb3 = combine_nodes(helper1, w1);
    auto comb4 = combine_nodes(helper1, w1);

    std::vector<ff_node*> firstSet={ &comb1, &comb2, &comb3, &comb4 };

    assert(firstSet.size() == NWORKER1);
    
    const Worker2 w2;
    std::vector<ff_node*> secondSet={
        new ff_comb(helper2,w2),
        new ff_comb(helper2,w2),
        new ff_comb(helper2,w2)
    };

    assert(secondSet.size() == NWORKER2);
    ff_a2a a2a;
    a2a.add_firstset(firstSet, 1000, false);
    a2a.add_secondset(secondSet, true);

    a2a.wrap_around();  // adding feedback channels

    
    // ---- last stage (Collector)
    Last last;

    // ---- building the topology
    ff_farm farm;
    farm.add_emitter(&first);
    farm.add_collector(&last);
    farm.add_workers({&a2a});

    if (farm.run_and_wait_end() <0) {
        error("running pipe\n");
        return -1;
    }
    return 0;
}
