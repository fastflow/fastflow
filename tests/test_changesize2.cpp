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

/*  pipe( First, A2A, Last )
 *                  
 *                  |-> Worker1  ->|
 *                  |              | -> Worker2 -->|  
 *                  |-> Worker1  ->|               | 
 *     First ---->  |              | -> Worker2 -->| -----> Last
 *                  |-> Worker1  ->|               | 
 *                  |              | -> Worker2 -->| 
 *                  |-> Worker1  ->|
 *
 */

/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

const int NTASKS = 500;
const int NWORKER1 = 4;
const int NWORKER2 = 3;

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
        int nworkers = this->get_num_outchannels();
        for(int k=0;k<NTASKS;++k) {
            for(int i=0;i<nworkers; ++i)
                ff_send_out_to(new long(k), i);
        }
        return EOS;
    }

};


struct MultiInputHelper: ff_minode_t<long> {
	long *svc(long *in) { return in; }
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
        if (get_my_id()==1) usleep(4000);
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
        size_t oldsz;
        for(size_t i=0;i<w.size(); ++i)
            w[i]->change_inputqueuesize(1, oldsz);   
        return 0;        
    }
    long* svc(long* in) {
        if (get_my_id() == 0) usleep(5000);
        return in;
    }
};

struct Last: ff_minode_t<long> {
    long* svc(long* in) {
        ++cnt;
        delete in;
        return GO_ON;
    }
    void svc_end() {
        if (cnt != NTASKS*NWORKER1) {
            std::cerr << "Error received " << cnt << " instead of " << NTASKS*NWORKER1 << "\n";
            exit(-1);
        }
    }
    
    long cnt=0;
};

int main() {
    // ---- first stage
    First first;

    // ----- building all-to-all

    const MultiInputHelper  helper;
    const MultiInputHelper2 helper2;
    const Worker1 w1;
    auto comb1 = combine_nodes(helper, w1);
    auto comb2 = combine_nodes(helper, w1);
    auto comb3 = combine_nodes(helper, w1);
    auto comb4 = combine_nodes(helper, w1);

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

    // ---- last stage
    Last last;

    // ---- building the topology
    ff_Pipe<> pipe(first, a2a, last);

    if (pipe.run_and_wait_end() <0) {
        error("running pipe\n");
        return -1;
    }
    return 0;
}
