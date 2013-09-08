/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

#include "ff/farm.hpp"
using namespace std;
using namespace ff;
#define NWORKERS 4
#define INBUF_Q_SIZE 4
//#define MNODES 2
#define CORES 4

typedef struct ff_task {
    int sourceW;
    int mnode;
	int core;
	int done;
} ff_task_t;

class emitter: public ff_node {
public:
	emitter(ff_loadbalancer* const lb_, ff_node** workers_) {
		lb = lb_;
		workers = workers_;
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
                    int targetworker = workers[i]->get_my_id();                 
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
                    bool res = lb->ff_send_out_to(worker_task, targetworker, 0 /* retry - total */);
                    if (res) 
                        printf("sent to worker %d on core %d mnode %d\n",
                               worker_task->sourceW,
                               worker_task->core,
                               worker_task->mnode);
                    else printf("ERROR: send failed - should never happen - task is lost - queue are too short\n");
                }
            }
		} else {		  
            if (worker_task->done<10) {
		   
                printf("[E] recv from worker %d on core %d mnode %d\n",
                       worker_task->sourceW, worker_task->core,
                       worker_task->mnode);
                lb->ff_send_out_to(worker_task, worker_task->sourceW);
            } else {
                completed[worker_task->sourceW] = true;
            }
		}
		/* usi male il GO_ON - brutto il for - basta contare
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
    ff_node** workers;
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
		printf("[%d] received from emitter task workerid %d on core %d\n",
		       get_my_id(), worker_task->sourceW, worker_task->core);
		//ff_send_out(worker_task); //inutile
		// return GO_ON ; // no
        ++taskcount;
        return (worker_task);
	}
    void svc_end (){
        printf("[%d] processed %d tasks\n",get_my_id(),taskcount);
    }
private:
    int taskcount;
};

int main() {
	std::vector<ff_node *> workers;
	ff_farm<> farm(false);
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

