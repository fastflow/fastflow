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

/**
 *
 * This is a basic test for testing the capability of starting and stopping 
 * OFarm's workers within the load-balancer thread.
 *
 */
#include <vector>
#include <cstdio>
#include <ff/farm.hpp>

using namespace ff;

class Emitter: public ff_node_t<long> {
protected:
    void freezeAll() {
        lb->freezeWorkers();
        gt->freeze();
    }
    void blockAll() {
        lb->broadcast_task(EOSW); 
        lb->wait_freezingWorkers();
        gt->wait_freezing();
    }
    void wakeupAll(bool freeze=true) {
        lb->thawWorkers(freeze, -1);  // wake-up all workers
        gt->thaw(freeze, -1);         // wake-up the collector
    }
    void stopAll() {
        lb->stop();
        gt->stop();
        wakeupAll(false);
    }
public:
    Emitter(long streamlen, 
            ff_loadbalancer *const lb, ff_gatherer *const gt):streamlen(streamlen), lb(lb), gt(gt) {}
    

    int svc_init() {     
        freezeAll();    // set freezing flag to all workers
        blockAll();     // wait that all workers go to sleep
        return 0;
    }

    long *svc(long *) {
        
        for(int i=0;i<10 && streamlen>0; ++i) {
            printf("iter %d\n", i);
            usleep(i*10000);

            // restart all workers
            wakeupAll();
            
            if (streamlen>0) {
                for(long k=1;k<=111;++k)
                    ff_send_out(new long(streamlen-k));
                streamlen-=111;
            }
                            
            // put workers to sleep
            blockAll();
        }
        // restart workers before sending EOS
        stopAll();

        return EOS;
    }
private:
    long                   streamlen;
    ff_loadbalancer *const lb;
    ff_gatherer     *const gt;
};

class Collector: public ff_node_t<long> {
public:
    Collector(long streamlen):expected(streamlen),error(false) {}

    long* svc(long *task) {
        const long &t = *task;
        
        if (t != --expected) {
            printf("ERROR: task received out of order, received %ld expected %ld\n", t, expected);
            error = true;
        }
                
        return GO_ON;
    }
    void svc_end() {
        if (error) abort();
    }
private:
    long expected;
    bool error;
};



class Worker:public ff_node_t<long> {
public:
    int svc_init() {
        printf("worker%ld (%ld) woken up\n", get_my_id(), getOSThreadId());
        return 0;
    }
    long *svc(long *t) {
        printf("worker%ld received %ld\n", get_my_id(), *t);
        return t;
    }
    void svc_end() {
        printf("worker %ld going to sleep\n", get_my_id());
    }
};

int main(int argc, char *argv[]) {
    int nworkers = 3;
    if (argc>1) {
        if (argc!=2) {
            printf("use: %s nworkers\n",argv[0]);
            return -1;
        }
        
        nworkers = atoi(argv[1]);
    }
    ff_OFarm<> ofarm([&]() {            
            std::vector<std::unique_ptr<ff_node> > W;
            for(int i=0;i<nworkers;++i)
                W.push_back(make_unique<Worker>());
            return W;
        } () );
    Emitter E(10000, ofarm.getlb(), ofarm.getgt());
    Collector C(10000);
    ofarm.setEmitterF(E);
    ofarm.setCollectorF(C);
    ofarm.setFixedSize(true);
    ofarm.setInputQueueLength(4); // setting very small queues
    ofarm.setOutputQueueLength(4);

    ofarm.run();
    ofarm.getgt()->wait(); // waiting for the termination of the collector
    return 0;
}
