#ifndef _CWC_FF_STAT_ACCEL_HPP_
#define _CWC_FF_STAT_ACCEL_HPP_

#include <ff/farm.hpp>
#include <ff/node.hpp>
using namespace ff;

#include <ff_definitions.hpp>
#include <statistics.hpp>

/*
  ---------------------------------------------------------
  StatFarm_Worker:
  - compute/write statistics over a window (single monitor)
  ---------------------------------------------------------
*/
class StatFarm_Worker: public ff_node {
public:
  StatFarm_Worker(vector<vector<Statistical_Engine<multiplicityType> *> > *engines) :
    engines(*engines) {}

  void * svc(void * task) {
    stat_task_t *t = (stat_task_t *)task;
    Stat_Window<multiplicityType> &window = *t->window;
    unsigned int mid = t->mid;
    unsigned int sid = t->sid;
    FREE(t);
    engines[mid][sid]->compute_write(window);
    return GO_ON;
  }

  //private:
  vector<vector<Statistical_Engine<multiplicityType> *> > &engines;
};



/*
  ----------------------------------------
  StatFarm_Emitter:
  - (monitor,engine)-to-worker n:1 binding
  ----------------------------------------
*/
// You have to extend the ff_loadbalancer....
class my_loadbalancer: public ff_loadbalancer {
protected:
  // implement your policy...
  inline size_t selectworker() { return victim; }

public:
  // this is necessary because ff_loadbalancer has non default parameters....
  my_loadbalancer(int max_num_workers):ff_loadbalancer(max_num_workers) {}

  void set_victim(int v) { victim=v;}

private:
  size_t victim;
};

// emitter filter
class StatFarm_Emitter: public ff_node {
public:
  StatFarm_Emitter(int nworkers, int n, my_loadbalancer * const lb):
    nworkers(nworkers), n(n), lb(lb) {}

  void * svc(void * task) {
    if (task == NULL) {
      for(int i=0;i<nworkers;++i) {
	lb->set_victim(i);
	ff_send_out(task);
      }
      return NULL;
    }
    
    stat_task_t *t = (stat_task_t *)task;
    lb->set_victim((t->sid * n + t->mid) % nworkers); /* set next worker to select */
    ff_send_out(t);
    
    return GO_ON;
  }
    
private:
  int nworkers, n;
  my_loadbalancer * lb;
};
#endif
