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
 *          farm1                     feedback( pipeline( feedback(farm2-master-worker),   A2A) )
 *                               ____________________________________________________________________
 *                              |                                                                    |
 *                              |                                                    |-->Second-->|  |
 *         |-->Worker1-->|      v       |--> Worker2 -->|               |-->First -->|            |  |
 *  Gen -->|             |--> Emitter-->|            |  |-->Collector-->|            |-->Second-->|--
 *         |-->Worker1-->|      ^ ^     |--> Worker2 -->|      |        |-->First -->|            |
 *                              | |__________________|         |                     |-->Second-->|
 *                              |______________________________|                             
 * 
 *                          |<- multi->|
 *                              input
 *                          |<--- master-worker ----->| |<------- all-to-all -------->|
 * |<-farm (no collector)->||<----- pipeline of master-worker farm and all-to-all ----->|
 *                              
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
        for(long i=1;i<=ntasks;++i) {
            ff_send_out((long*)i);
        }
        return EOS;
    }
    const long ntasks=1000;
};

struct Worker1: ff_node_t<long> {
    long* svc(long* in) {
        return in;
    }    
};
struct Worker2: ff_monode_t<long> {
    long* svc(long* in) {
        ff_send_out_to(in, 0);
        ff_send_out_to(in, 1);
        return GO_ON;
    }    
};
struct Collector: ff_monode_t<long> {
    long* svc(long* in) {
        ff_send_out_to(in, 0);  // backward
        ff_send_out_to(in, next + 1);  // forward
        next = ((next + 1) % (get_num_outchannels() - 1)); // get_num_outchannels() comprises also feedback channels
        return GO_ON;
    }
    void eosnotify(ssize_t) {
        printf("Collector received EOS\n");
    }
        
    
    int next = 0;
};

struct PipeInternal: ff_pipeline {
    struct Emitter: ff_minode_t<long> {
        Emitter(size_t nworkers):nworkers(nworkers) {}
        long* svc(long* in) {
            size_t channelid = get_channel_id();
            if (fromInput()) {
                printf("received %ld from Worker1 (%ld)\n", (long)in, channelid);
                ++ntasks;
                return in;
            }
            if (channelid < nworkers) { // from workers
                printf("received %ld from Worker2 (%ld)\n", (long)in, channelid);
                return GO_ON;
            } else {
                if (channelid == nworkers) { // from Collector
                    printf("received %ld from Collector\n", (long)in);
                    return GO_ON;
                }
            }
            
            printf("received %ld from Second (%ld)\n", (long)in, channelid);
            --ntasks;
            if (ntasks==0 && eosreceived) {
                printf("SENDING EOS\n");
                return EOS;
            }
            return GO_ON;
        }

        void eosnotify(ssize_t) {
            if (++neos == get_num_inchannels()) {
                eosreceived = true;
                if (ntasks==0) ff_send_out(EOS);
            }
        }
        const size_t nworkers;
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
            printf("Second (%ld) sending back %ld\n", get_my_id(), (long)in);
            return in;
        }
        void eosnotify(ssize_t) {
            printf("Second (%ld) received EOS\n", get_my_id());
        }
    };
    
    PipeInternal() {
        const size_t nworkers=3;   
        ff_farm* farm = new ff_farm;
        ff_comb* comb = new ff_comb(new Emitter(nworkers), new EmitterHelper, true, true);
        farm->add_emitter(comb);
        farm->cleanup_all();
        std::vector<ff_node*> W;
        for(size_t i=0;i<nworkers;++i)
            W.push_back(new Worker2);
        farm->add_workers(W);
        farm->wrap_around();
        farm->add_collector(new Collector);
        farm->wrap_around();
        
        ff_a2a* a2a = new ff_a2a;
        std::vector<ff_node*> W1;
        W1.push_back(new First);
        W1.push_back(new First);
        a2a->add_firstset(W1,0, true);
        
        std::vector<ff_node*> W2;
        W2.push_back(new Second);
        W2.push_back(new Second);
        a2a->add_secondset(W2,true);

        this->add_stage(farm, true);
        this->add_stage(a2a,  true);
        this->wrap_around();
    }
};


int main() {
    std::vector<ff_node*> W;
    W.push_back(new Worker1);
    W.push_back(new Worker1);
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
