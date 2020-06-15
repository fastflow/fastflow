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
 *  Please see other tests that do not use condition variable explicitly.
 *
 */

#ifndef __APPLE__
#include <ff/platforms/platform.h>
#endif
#include <iostream>
#include <ff/ff.hpp>
#include <ff/allocator.hpp>


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
        ticks_wait(task->iter);

        return t;
    }

};



class Emitter: public ff_monode {
public:
    Emitter(int ntasks,int nworkers):
        ntasks(ntasks),nworkers(nworkers),getback(0) {}
    
    void ff_sendout(int idx, fftask_t* task) {
        if (idx<0) { 
            ff_send_out((void*)task); 
            return; 
        }        
        ff_send_out_to((void*)task,idx);        
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
        broadcast_task(new fftask_t(WAIT, 0));

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
};


int main(int argc, char* argv[]) {    
#if defined(BLOCKING_MODE)
    printf("TODO: mixing dynamic behavior and blocking mode has not been tested yet!!!!!\n");
    return 0;
#endif

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
    
    ff_farm farm;
    Emitter emitter(ntask,nworkers);
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

