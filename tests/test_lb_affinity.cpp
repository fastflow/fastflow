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

#if !defined(FF_INITIAL_BARRIER)
// to run this test we need to be sure that the initial barrier is executed
#define FF_INITIAL_BARRIER
#endif
#include <ff/ff.hpp>

using namespace ff;

#define NWORKERS 4
#define INBUF_Q_SIZE 4

//#define MNODES 2
#define CORES 4

typedef struct ff_task {
    size_t sourceW;
    size_t mnode;
	size_t core;
	size_t done;
} ff_task_t;

class emitter: public ff_node {
public:
	emitter(ff_loadbalancer* const lb, const svector<ff_node*> &workers):
        lb(lb),workers(workers) {
		completed = new bool[NWORKERS];
		for (int i = 0; i < NWORKERS; i++)
			completed[i] = false;
        done=0;
	} 

	void* svc(void *t) {
		ff_task_t* worker_task = (ff_task_t*) t;
		if (worker_task == NULL) {
            for (int j = 0; j < INBUF_Q_SIZE; j++) {
                for (int i = 0; i < NWORKERS; i++) {
                    size_t targetworker = workers[i]->get_my_id();
                    //
                    // NOTE: in order to use the following low-level call, you must be sure that the 
                    //       threads are started. To do this you can compile with FF_INITIAL_BARRIER
                    //
                    int targetcore = threadMapper::instance()->getCoreId(lb->getTid(workers[i]));
                    int targetmnode = targetcore/8;
                    // here allocate the task
                    // numa_malloc(sizeof(ff_task_t),mnode)
                    worker_task = (ff_task_t *) malloc(sizeof(ff_task_t));
                    worker_task->sourceW = targetworker;
                    // here danger of race condition
                    // only static information can be used
                    worker_task->core = targetcore;
                    worker_task->mnode = targetmnode;
                    worker_task->done = 0;
                    bool res = lb->ff_send_out_to(worker_task, targetworker);
                    if (res) 
                        printf("sent to worker %ld on core %ld mnode %ld\n",
                               worker_task->sourceW,
                               worker_task->core,
                               worker_task->mnode);
                    else printf("ERROR: send failed - should never happen - task is lost - queue are too short\n");
                }
            }
		} else {		  
            if (worker_task->done<10) {
		   
                printf("[E] recv from worker %ld on core %ld mnode %ld\n",
                       worker_task->sourceW, worker_task->core,
                       worker_task->mnode);
                lb->ff_send_out_to(worker_task, worker_task->sourceW);
            } else {
                completed[worker_task->sourceW] = true;
            }
		}
		/* 
           for (int i = 0; i < NWORKERS; i++)
           if (!completed[i])
             return GO_ON ;
		*/
		for (int i = 0; i < NWORKERS; i++)
            done &= completed[i];
        if (done) {
                free(worker_task);
                delete[] completed;
                return NULL;
            }
        else done = true;
        return GO_ON;
	}
  
private:
    ff_loadbalancer* lb;
    const svector<ff_node*> &workers;
    bool* completed;
    bool done;  
};

class worker: public ff_node {
public:
    worker (): taskcount(0){};
	void* svc(void* t) {
		ff_task_t* worker_task = (ff_task_t*) t;
        ++worker_task->done;
		usleep(worker_task->sourceW * 100);
		printf("[%ld] received from emitter task workerid %ld on core %ld\n",
		       get_my_id(), worker_task->sourceW, worker_task->core);
		//ff_send_out(worker_task); //inutile
		// return GO_ON ; // no
        ++taskcount;
        return (worker_task);
	}
    void svc_end (){
        printf("[%ld] processed %d tasks\n",get_my_id(),taskcount);
    }
private:
    int taskcount;
};

int main() {
	std::vector<ff_node *> workers;
	ff_farm farm(false);
	for (int i = 0; i < NWORKERS; i++)
		workers.push_back(new worker());
	emitter em(farm.getlb(), farm.getWorkers());
	farm.add_emitter(&em);
	farm.add_workers(workers);
	farm.set_scheduling_ondemand(INBUF_Q_SIZE);
	farm.wrap_around();
	if (farm.run_and_wait_end() < 0) {
		error("running farm\n");
		return -1;
	}
	return 0;
}

