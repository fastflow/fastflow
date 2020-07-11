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
/* Building blocks test:
 *
 *      pipe( farm1(Worker), feedback(pipe(farm2(A2A(First,Second)))) )
 *
 *                                ______________________________________
 *                               |                                      |
 *                               |                     |-->Second-->|   |
 *         |-->Worker-->|        v        |-->First -->|            |   |
 *  Gen -->|            | --> Emitter --->|            |-->Second-->|---|
 *         |-->Worker-->|                 |-->First -->|            |
 *                                                     |-->Second-->|
 *                              ^
 *                              |____ farm's scheduler with multi-input filter
 * 
 *                                      |<------- all-to-all -------->|
 * |<-farm (no collector)->| |<--- pipeline of farm (no collector) ---->|
 *                                 the farm's worker is an A2A
 *
 */
/*
 * Author: Massimo Torquati
 *
 */

#include <ff/ff.hpp>

using namespace ff;

struct Generator: ff_node_t<long> {
    long* svc(long*) {
        for(long i=1;i<=10;++i) {
            ff_send_out((long*)i);
        }
        return EOS;
    }
};

struct Worker: ff_node_t<long> {
    long* svc(long* in) {
        return in;
    }    
};

struct PipeInternal: ff_pipeline {
    struct Emitter: ff_minode_t<long> {
        long* svc(long* in) {
            ssize_t channelid = get_channel_id();
            if (fromInput()) {
                printf("received from Worker%ld\n", channelid);
                ++ntasks;
                return in;
            }
            printf("received back from Second%ld\n", channelid);
            --ntasks;
            if (ntasks==0 && eosreceived) return EOS;
            return GO_ON;
        }

        void eosnotify(ssize_t) {
            if (++neos == get_num_inchannels()) {
                eosreceived = true;
                if (ntasks==0) ff_send_out(EOS);
            }
        }
        size_t neos=0;
        long ntasks=0;
        bool eosreceived=false;
    };
    struct EmitterHelper: ff_monode_t<long> {
        long *svc(long*in) {return in;}
        void eosnotify(ssize_t) { broadcast_task(EOS); }
    };
    struct First: ff_node_t<long> {
        long* svc(long* in) {
            return in;
        }    
    };
    struct Second: ff_node_t<long> {
        long* svc(long* in) {
            printf("Second%ld sending back %ld\n", get_my_id(), (long)in);
            return in;
        }    
    };
    
    PipeInternal() {
        ff_farm* farm = new ff_farm;
        ff_comb* comb = new ff_comb(new Emitter, new EmitterHelper, true, true);
        farm->add_emitter(comb);
        farm->cleanup_all();

        ff_a2a* a2a = new ff_a2a;
        std::vector<ff_node*> W1;
        W1.push_back(new First);
        W1.push_back(new First);
        a2a->add_firstset(W1,0, true);
        
        std::vector<ff_node*> W2;
        W2.push_back(new Second);
        W2.push_back(new Second);
        a2a->add_secondset(W2,true);

        std::vector<ff_node*> W;
        W.push_back(a2a);
        farm->add_workers(W);
        farm->wrap_around();
        this->add_stage(farm, true);
    }
    void *svc(void*) { abort();}
};


int main() {
    std::vector<ff_node*> W;
    W.push_back(new Worker);
    W.push_back(new Worker);
    ff_farm farm(W, new Generator);
    farm.remove_collector();
    farm.cleanup_all();
    
    PipeInternal pipeI;

    ff_Pipe<> pipe(farm, pipeI);
    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }

    return 0;
}
