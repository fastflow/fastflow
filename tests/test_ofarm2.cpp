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
 *   Just one single ordered farm (building block) with auto-scheduling policy.
 *               
 *           --> Worker1 -->         
 *          |               |        
 * Start --> --> Worker1 -->  ---> Stop
 *          |               |        
 *           --> Worker1 -->         
 *                                     
 *  By defining TEST_BROADCAST it is possible to test broadcast_task and all_gather with 
 *  ordered farm!               
 */


#include <vector>
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

class Start: public ff_monode_t<long> {
public:
    Start(long streamlen):streamlen(streamlen) {}
    long* svc(long*) {    
        if (--streamlen) {
#if !defined(TEST_BROADCAST)
            ff_send_out((long*)streamlen);
#else            
            broadcast_task((long*)streamlen);
#endif            
            return GO_ON;
        }
        return EOS;
    }
private:
    long streamlen;
};


class Worker1: public ff_node_t<long> {
public:
    long * svc(long * task) {
        printf("Worker received task %ld\n", (long)task);
        if (get_my_id() == 0) usleep(random() % 20000);
        return task;
    }
};


class Stop: public ff_minode_t<long> {
public:
    Stop(long streamlen):
        expected(streamlen),error(false) {}

    long* svc(long* task) {    
        long t = (long)task;
        printf("received %ld from %ld\n", t, get_channel_id());

#if !defined(TEST_BROADCAST)        
        if (t != --expected) {
            printf("ERROR: task received out of order, received %ld expected %ld\n", t, expected);
            error = true;
        }
#else
        --expected;
        std::vector<long*> V;
        all_gather(task,V);
        for(size_t i=0;i<V.size();++i) {
            auto t = (long)V[i];
            if (t != expected) {
                printf("ERROR: task received out of order, received %ld expected %ld\n", t, expected);
                error = true;
            }
        }

#endif
        return GO_ON;
    }
    void svc_end() {
        if (error) abort();
    }
private:
    long expected;
    bool error;
};


int main(int argc, char * argv[]) {
    int nworkers = 3;
    long streamlen = 1000;
    if (argc>1) {
        if (argc<3) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nworkers streamlen\n";
            return -1;
        }
               
        nworkers=atoi(argv[1]);
        streamlen=atol(argv[2]);
    }

    if (nworkers<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    srandom(131071);
        
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker1);
    ff_farm ofarm(w, new Start(streamlen), new Stop(streamlen));
    ofarm.set_ordered();               // ordered farm
    ofarm.set_scheduling_ondemand();   // auto-scheduling policy 
    ofarm.cleanup_all();

    if (ofarm.run_and_wait_end()<0) {
        error("running ofarm\n");
        return -1;
    }

    std::cerr << "DONE\n";
    return 0;
}
