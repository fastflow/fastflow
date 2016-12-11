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
 *  This is a dummy example on how to put workers to sleep for a while.
 *
 */

#include <ff/platforms/platform.h>
#include <vector>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>
#include <ff/node.hpp>
#include <ff/allocator.hpp>
#include <iostream>

using namespace ff;

typedef enum {WAIT=1, WORK} op_t;

struct fftask_t {
    fftask_t(op_t op, unsigned iter):op(op),iter(iter) {}
    op_t     op;
    unsigned iter;
};

/* these mutex and cond variables are globals but they shoud be x thread 
 * Dynamic initialisation for WIN compatibility
 * 
 */
pthread_mutex_t mutex; // = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond; // = PTHREAD_COND_INITIALIZER;
static bool waiting=false;


void WAIT_SIGNAL() {
    pthread_mutex_lock(&mutex);
    while(waiting) pthread_cond_wait(&cond, &mutex);
    pthread_mutex_unlock(&mutex);
}

void BCAST_SIGNAL() {
    pthread_mutex_lock(&mutex);
    pthread_cond_broadcast(&cond);
    waiting=false;
    pthread_mutex_unlock(&mutex);
}

class Worker: public ff_node {
public:
    void* svc(void* t) {
        fftask_t* task = (fftask_t*)t;
        if (task->op == WAIT) {
            printf("[%ld] I have to wait! Waiting to be woken up\n", get_my_id());
            WAIT_SIGNAL();
            printf("[%ld] I'm wide awake\n", get_my_id());
            return GO_ON;
        }
        
        printf("[%ld] I'm working\n", get_my_id());
        for(volatile unsigned i=0;i<task->iter;++i);
        
        return t;
    }

};


class myScheduler: public ff_loadbalancer {
protected:
    inline size_t selectworker() { 
        size_t sel = victim++ % getnworkers(); 
        printf("selected %zd\n", sel);
        return sel;
    }
public:
    myScheduler(int max_num_workers):ff_loadbalancer(max_num_workers) {
        victim=0;
    }
    void set_victim(int v) { victim=v; }
    void broadcast(void* task) { ff_loadbalancer::broadcast_task(task); }
private:
    int    victim;
};


class Emitter: public ff_node {
public:
    Emitter(int ntasks,int nworkers,myScheduler* lb):
        ntasks(ntasks),nworkers(nworkers),getback(0),lb(lb) {}
    
    void ff_sendout(int idx, fftask_t* task) {
        if (idx<0) { 
            ff_send_out((void*)task); 
            return; 
        }        
        lb->set_victim(idx);
        ff_send_out((void*)task);        
    }
    
    void* svc(void* t) {
        if (!t) {
            // send all task to the workers
            for(int i=0;i<ntasks;++i)
                ff_sendout(-1, new fftask_t(WORK,i));

            // put all workers but 0 and 1 to sleep
            waiting = true;
            for(int i=2;i<nworkers;++i) 
                ff_sendout(i, new fftask_t(WAIT,0));
            return GO_ON;
        }
        
        ++getback;
        if (getback<ntasks) return GO_ON;

        sleep(2);
        printf("checking now who is sleeping\n");
        lb->broadcast(new fftask_t(WAIT, 0));

        sleep(2);
        printf("waking up all threads now!\n");
        BCAST_SIGNAL();

        sleep(2);
        printf("ENDING\n");
        return NULL;               
    }
private:
    int ntasks;
    int nworkers;
    int getback;
    myScheduler* lb;
};


int main(int argc, char* argv[]) {    
	// Init global mutexes and cond vars
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);
    int nworkers = 3;
	int ntask = 1000;

    if (argc>1) {
        if (argc<3) {
            printf("use: %s nworkers ntask\n", argv[0]);
            return -1;
        }
    
        nworkers=atoi(argv[1]);
        if (nworkers<=2) {
            printf("for this test nworkers shoud be greater than 2\n");
            return -1;
        }
        
        ntask=atoi(argv[2]);
    }
    if (nworkers<=0 || ntask<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    ff_farm<myScheduler> farm;
    Emitter emitter(ntask,nworkers,farm.getlb());
    farm.add_emitter(&emitter);
    farm.set_scheduling_ondemand(2);

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);
    farm.add_collector(NULL);
    farm.wrap_around();
    farm.run_and_wait_end();

    std::cerr << "DONE\n";
    return 0;
}

