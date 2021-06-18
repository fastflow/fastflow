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
#include <ff/ff.hpp>

using namespace ff;

class Emitter: public ff_node_t<long> {
protected:
    void freezeAll() {
        ofarm->getlb()->freezeWorkers();
        ofarm->getgt()->freeze();
    }
    void blockAll() {
        ofarm->getlb()->broadcast_task(EOSW); 
        ofarm->getlb()->wait_freezingWorkers();
        ofarm->getgt()->wait_freezing();
    }
    void wakeupAll(bool freeze=true) {
        ofarm->getlb()->thawWorkers(freeze, -1);  // wake-up all workers
        ofarm->getgt()->thaw(freeze, -1);         // wake-up the collector
    }
    void stopAll() {
        ofarm->getlb()->stop();
        ofarm->getgt()->stop();
        wakeupAll(false);
    }
public:
    Emitter(long streamlen, 
            ff_farm* ofarm):streamlen(streamlen),ofarm(ofarm) {}
    
    int svc_init() {     
        freezeAll();    // set freezing flag to all workers
        blockAll();     // wait that all workers go to sleep
        return 0;
    }

    long *svc(long *) {

        for(int i=0;i<10; ++i) {
            printf("iter %d\n", i);
            usleep(i*10000);

            // restart all workers
            wakeupAll();
            
            for(long k=0;k<streamlen;++k)
                ff_send_out(new long(i*streamlen + k));
            
            // put workers to sleep
            blockAll();
        }
        // restart workers before sending EOS
        stopAll();

        return EOS;
    }
private:
    long      streamlen;
    ff_farm  *ofarm;
};

class Collector: public ff_minode_t<long> {
public:
    Collector(long): error(false) {}

    long* svc(long *task) {
        const long &t = *task;

        printf("Collector received %ld from %ld\n",t,get_channel_id());
        if (t != expected) {
            printf("ERROR: task received out of order, received %ld expected %ld\n", t, expected);
            error = true;
        }
        ++expected;
        return GO_ON;
    }
    void svc_end() {
        if (error) abort();
    }
private:
    long expected = 0;
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
        if (get_my_id()==0 || get_my_id()==1) usleep(5000);
        return t;
    }
    void svc_end() {
        printf("worker %ld going to sleep\n", get_my_id());
    }
};

int main(int argc, char *argv[]) {
#if defined(BLOCKING_MODE)
    printf("TODO: mixing dynamic behavior and blocking mode has not been tested yet!!!!!\n");
    return 0;
#endif

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
    Emitter E(10, &ofarm);
    Collector C(10);
    ofarm.add_emitter(E);
    ofarm.add_collector(C);
    ofarm.setInputQueueLength(4, true); // setting very small queues
    ofarm.setOutputQueueLength(4, true);
    ofarm.run();
    ofarm.getgt()->wait(); // waiting for the termination of the collector

    return 0;
}
