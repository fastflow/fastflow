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
 *  pipe(Farm1-nocollector, loopback(Farm2-nocollector))
 *
 *                                               ____________________
 *                                              |                    |
 *                     --> Worker1 -->          |       --> Worker2--|
 *                    |               |         v      |             ^
 *         Emitter1 -- --> Worker1 -->  --> Emitter2 -- --> Worker2--|                        
 *                    |               |         ^      |
 *                     --> Worker1 -->          |       --> Worker2--|
 *                                              |____________________|   
 *
 */

#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

class Emitter1: public ff_node_t<long> {
public:
    Emitter1(ff_loadbalancer *lb, int ntasks):lb(lb),ntasks(ntasks) {}
    long* svc(long*) {
        // enforces nonblocking mode since the beginning
        // regardless the compilation setting
        lb->broadcast_task(NBLK); 
        for(long i=0;i<ntasks;++i) {
            ff_send_out((long*)(i+10));
            switch(i) {
            case 100:{
            lb->broadcast_task(BLK);
            } break;
            case 500: {
            lb->broadcast_task(NBLK);
            } break;
            case 800: {
            lb->broadcast_task(BLK);
            } break;
            }
        }
        return EOS;
    }
private:
    ff_loadbalancer *lb;        
    const int ntasks;
};

class Emitter2: public ff_node_t<long> {
public:
    Emitter2(ff_loadbalancer *lb, long nworkers):neos(0),nworkers(nworkers), lb(lb) {}
    long* svc(long* t) {
        int wid = lb->get_channel_id();
        if (wid == -1) {
            printf("TASK FROM INPUT %ld\n", (long)(t));
            return t;
        }
        printf("got back a task from Worker2(%d)\n", wid);
        return GO_ON;
    }
    void eosnotify(ssize_t id) {
        if (id != -1) return;
        if (++neos == nworkers) lb->broadcast_task(EOS);        
    }
private:
    int  neos;
    long nworkers;
    ff_loadbalancer *lb;    
};

struct Worker1: ff_node_t<long> {
    long* svc(long* task) {
        return task;
    }
};

struct Worker2: ff_node_t<long> {
    long* svc(long* task) {
        return task;
    }
};


int main(int argc, char* argv[]) {
    size_t nworkers = 3;
    size_t ntasks = 1000;
    if (argc>1) {
        if (argc < 3) {
            std::cerr << "use:\n" << " " << argv[0] << " numworkers ntasks\n";
            return -1;
        }
        nworkers  =atoi(argv[1]);
        ntasks    =atoi(argv[2]);
    }

    ff_Farm<long>   farm1(  [nworkers]() { 
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nworkers;++i)  W.push_back(make_unique<Worker1>());
            return W;
        } () );
    Emitter1 E1(farm1.getlb(), ntasks);
    farm1.add_emitter(E1);
    farm1.remove_collector();
    farm1.setFixedSize(true);
    farm1.setInputQueueLength(nworkers*100);
    farm1.setOutputQueueLength(nworkers*100);


    ff_Farm<long>   farm2(  [nworkers]() { 
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nworkers;++i)  W.push_back(make_unique<Worker2>());
            return W;
        } () );
    Emitter2 E2(farm2.getlb(), nworkers);
    farm2.add_emitter(E2);
    farm2.remove_collector();
    farm2.wrap_around();

    ff_Pipe<> pipe(farm1, farm2);
    pipe.setXNodeInputQueueLength(100,true);
    pipe.setXNodeOutputQueueLength(100,true);
    pipe.run_and_wait_end();
    return 0;
}
