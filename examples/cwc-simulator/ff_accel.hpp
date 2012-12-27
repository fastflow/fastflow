/*
  This file is part of CWC Simulator.

  CWC Simulator is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  CWC Simulator is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with CWC Simulator.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _CWC_FFACCEL_HPP_
#define _CWC_FFACCEL_HPP_

//#include <ctime>
#include <sstream>
#include <deque>
#include "definitions.h"
#include "Simulation.h"
//#include "statistics.hpp"
#include <ff_definitions.hpp>
#include "utils.h"
#include <ff/platforms/platform.h>
#include <ff/farm.hpp>
#include <ff/mapping_utils.hpp>
/*
#include <ff/node.hpp>
#include <ff/lb.hpp>
*/
//#include <sys/time.h>
//#include <sys/resource.h>

using namespace ff;
using namespace std;
using namespace cwc_utils;

//speed comparison (the slower task has higher priority)
class simulation_task_comparison {
public:
  bool operator() (simulation_task_t * &lhs, simulation_task_t * &rhs) {
    // lhs > rhs (i. e. lhs has higher priority)
    // if lhs has produced less samples than rhs
    return lhs->n_sampled < rhs->n_sampled;
  }
};



class SimFarm_Worker: public ff_node {
public:
  SimFarm_Worker(
	 double t, //sampling period
	 int sps, //n. of samples per slice
	 int ns, //total n. of samples
	 int rank=-1 //thread-to-core mapping
	 ) :
    sampling_period(t), samples_per_slice(sps), n_samples(ns), rank(rank) {}

  int svc_init() {
    if(rank >= 0) {
      ff_mapThreadToCpu(rank);
      cerr << "(sim. farm-acc.) SimFarm_Worker pinned to core #" << rank << endl;
    }
    return 0;
  }

  void * svc(void * task) {
    simulation_task_t * t = (simulation_task_t *)task;
#ifdef LOG
    ofstream &logfile(t->simulation->get_logfile());
    logfile << "*** RESUME on worker {" << this << "}" << endl;
#endif
    int n_sampled = t->n_sampled; //n. of covered samples
    double next_sample_time = n_sampled * sampling_period; //first uncovered sample
    double tau = t->simulation->restore(); //restore the last tau computed
    double next_reaction_time = t->simulation->get_time() + tau;
    bool stall = t->simulation->get_stall();
    int next_sampling_stop = t->to_end ? n_samples : min(n_sampled + samples_per_slice, n_samples);

    //start timer
    double start_time = get_time_from(0, usage);
    double running_time;



    //*** start sampling engine
    while(true) {
#ifdef SIMLOG
      logfile << "last reaction: " << (t->simulation->get_time()) << "; "
	      << "next sample: " << next_sample_time << "; next reaction: " << next_reaction_time;
#endif
      if(next_reaction_time > next_sample_time || stall) {
	//monitor over the current term
#ifdef SIMLOG
	logfile << " -> monitor" << endl;
#endif

#ifdef HYBRID
	if(!stall)
	  //ODE from current ode-time to next_sample_time (table!)
	  t->simulation->ode(next_sample_time);
#endif

	sample_t *monitor = t->simulation->monitor();
	++n_sampled;

	//send output
	void *p = MALLOC(sizeof(out_task_t));
	ff_send_out((void *)new (p) out_task_t(t, next_sample_time, monitor)); //send out the sample
#ifdef LOG
	logfile << "*** sampling at time " << next_sample_time << endl;
#endif
	if(n_sampled == next_sampling_stop) {
	  //backup the last tau computed
	  t->simulation->freeze(tau);
#ifdef LOG
	  logfile << "--- END OF THE SLICE ---\n";
#endif
	  running_time = get_time_from(start_time, usage);
	  t->n_sampled = n_sampled;
	  break;
	}
	else
	  next_sample_time = n_sampled * sampling_period;
      }
      else {
	//fire the next reaction
#ifdef SIMLOG
	logfile << " -> fire" << endl;
#endif
	t->simulation->step(tau);
	tau = t->simulation->next_simulation_tau();
	if(tau != -1) {
	  next_reaction_time += tau;
	  //compute updates
	  t->simulation->compute_updates();
#ifdef HYBRID
	  //compute ode-delta table from 0 to tau
	  t->simulation->compute_ode_delta(tau);
#endif
	}
	else {
	  stall = true;
	  t->simulation->set_stall();
	}
      }
    }

#ifdef LOG
    logfile << "*** PAUSE" << endl;
#endif
    void *p = MALLOC(sizeof(out_task_t));
    t->running_time += running_time;
    return ((void *)new (p) out_task_t(t, t->running_time)); //ack
  }
  //*** end sampling engine

private:
  double sampling_period;
  int samples_per_slice;
  int n_samples;
  int rank;
  struct rusage usage;
};


class SimpleEmitter: public ff_node {
public:
  SimpleEmitter(int r) : rank(r) {}

  int svc_init() {
    if(rank >= 0) {
      ff_mapThreadToCpu(rank);
      cerr << "(Sim. farm-acc.) SimpleEmitter pinned to core #" << rank << endl;
    }
    return 0;
  }

  void *svc(void *task) {
    return task;
  }

private:
  int rank;
};

class SimpleCollector: public ff_node {
public:
  SimpleCollector(int r) : rank(r) {}

  int svc_init() {
    if(rank >= 0) {
      ff_mapThreadToCpu(rank);
      cerr << "(Sim. farm-acc.) SimpleCollector pinned to core #" << rank << endl;
    }
    return 0;
  }
  
  void *svc(void *task) {
    return task;
  }

private:
  int rank;
};
#endif
