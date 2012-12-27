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

#ifndef _CWC_FF_NODES_HPP_
#define _CWC_FF_NODES_HPP_

#include <queue>
#include <algorithm>
using namespace std;

#include <ff/node.hpp>
#include <ff/mapping_utils.hpp>

#include <random.h>
#include <Driver.h>
#include <Simulation.h>
#include <parameters.hpp>
#include <ff_definitions.hpp>
#include <utils.h>
//#include <sys/time.h>
//#include <sys/resource.h>
#ifdef USE_FF_ACCEL
#include <ff_accel.hpp>
#endif
#include <ProgressBars.hpp>
#include "statistics.hpp"


using namespace ff;

/*
  ------------------------------------------------------------------------------
  TasksGenerator:
  . produce seeds
  - parse input model (and build model-objects)
  - prepare and send sim. tasks
  ------------------------------------------------------------------------------
*/
class TasksGenerator: public ff_node {
public:
  TasksGenerator(cwc_parameters_t &param, int rank = -1) :
    param(param),
    n_simulations(param.n_simulations),
    fixed_seed(param.fixed_seed),
    fixed_seed_value(param.fixed_seed_value),
    fname(param.infname),
    rank(rank)
  {}

  int svc_init() {
    if(rank >= 0) {
      ff_mapThreadToCpu(rank);
      cerr << "TasksGenerator pinned to core #" << rank << endl;
    }
    return 0;
  }
    
  void * svc(void* task) {
    //seeds generator
    rng_b_type seed_rng(fixed_seed ? fixed_seed_value : (unsigned int)time(0));
    uint_gm_type seed_gm(0, (numeric_limits<int>::max)());
    uint_vg_type seed_vg(seed_rng, seed_gm);

    //parse
    scwc::Driver driver;
    ifstream *infile = new ifstream(fname.c_str());
    driver.parse_stream(*infile, fname.c_str());
    delete infile;

    //prepare sim. tasks and send them out
    void *p = MALLOC(sizeof(vector<simulation_task_t *>(n_simulations)));
    vector<simulation_task_t *> *tasks = new (p) vector<simulation_task_t *>(n_simulations);
    for(unsigned int i=0; i<n_simulations; i++) {
      int sim_seed = seed_vg(); //get a seed
      Simulation *fsp;
#ifdef HYBRID
	  MYNEW(fsp, Simulation, i, *driver.model, sim_seed
		, param.rate_cutoff
	    , param.population_cutoff
	    , param.sampling_period);
#else
	  MYNEW(fsp, Simulation, i, *driver.model, sim_seed);
#endif
	 
      p = MALLOC(sizeof(simulation_task_t));
      tasks->at(i) = new (p) simulation_task_t(fsp);
    }

#ifdef LOG
    cerr << "sending out sim. tasks" << endl;
#endif
    ff_send_out(tasks);
    return (void *)FF_EOS;
  }

private:
  cwc_parameters_t &param;
  unsigned int n_simulations;
  bool fixed_seed;
  int fixed_seed_value;
  string &fname;
  int rank;
};




/*
  ------------------------------------------------------------------------------
  SimEngine:
  - run simulations (scheduling etc.)
  - send out samples
  ------------------------------------------------------------------------------
*/
class SimEngine: public ff_node {
public:
  SimEngine(
	    const cwc_parameters_t &param,
#ifdef USE_FF_ACCEL
	    unsigned int samples_per_slice,
#endif
	    unsigned int n_samples, unsigned int n_monitors, int rank = -1
	    ) :
    n_monitors(n_monitors),
    sampling_period(param.sampling_period),
    time_limit(param.time_limit),
    n_samples(n_samples),
#ifdef USE_FF_ACCEL
    n_workers(param.n_workers),
    samples_per_slice(samples_per_slice),
    p_inflight(param.p_inflight),
#endif
    rank(rank)
  {
#ifdef DATASIZE
    datasize = 0;
    ixtime = -1;
#endif
  }

  int svc_init() {
    if(rank >= 0) {
      ff_mapThreadToCpu(rank);
      cerr << "SimEngine pinned to core #" << rank << endl;
    }
    return 0;
  }

  void* svc(void* task) {
    vector<simulation_task_t *> &tasks = *((vector<simulation_task_t *> *) task);
    n_simulations = tasks.size();
#ifdef TIME
    running_times = vector<double>(n_simulations, 0.0);
#endif
#ifdef GRAIN
    cycle_ticks = vector<double>(n_simulations, 0.0);
    steps = vector<unsigned int>(n_simulations, 0);
#endif

#ifdef USE_FF_ACCEL
    /*
      ----------------------------------
      SIM. FARM starts here
      ----------------------------------
    */
    ff_farm<> farm(true, n_simulations);  
    vector<ff_node *> workers;
    for(int i=0; i<n_workers; i++) {
      workers.push_back(new SimFarm_Worker(sampling_period, samples_per_slice, n_samples,
					   SIMFARM_WORKERS_RANK>=0? SIMFARM_WORKERS_RANK + i : -1));
    }
    farm.add_workers(workers);
#ifdef LOG
    cerr << "Farm ready to start\n";
#endif
    
    int n_active_tasks = n_simulations;
    int n_running_tasks = 0;
    priority_queue<simulation_task_t *, vector<simulation_task_t *>, simulation_task_comparison> ready_queue;

    SimpleEmitter *emitter = new SimpleEmitter(SIMFARM_EMITTER_RANK);
    SimpleCollector *collector = new SimpleCollector(SIMFARM_COLLECTOR_RANK);
    farm.add_emitter(emitter);
    farm.add_collector(collector);
    farm.set_scheduling_ondemand();
		
    //fill the ready-queue
    for(int i=0; i<n_simulations; i++) {
      ready_queue.push(tasks[i]);
    }

    //run
#ifdef LOG
    cerr << "Farm acc. starting" << endl;
#endif
    ProgressBars bars(n_simulations);
    farm.run();

    /*
      scheduling loop starts here
    */
    while(n_active_tasks) {
#ifdef LOG
      cerr << "---------------" << endl << "scheduling step" << endl
	   << "* jobs in RQ: " << ready_queue.size() << endl;
#endif
      //offload RQ
      while(!ready_queue.empty()) {
	simulation_task_t *t = ready_queue.top();
	ready_queue.pop();
	farm.offload(t);
#ifdef LOG
	cerr << "offloaded sim. #" << t->simulation->get_id();
	cerr << " (speed: " << t->n_sampled << " samples)" << endl;
#endif
      }
      //(active) waiting for some tasks
      n_running_tasks = p_inflight * n_active_tasks / 100;
      int n_to_collect = n_active_tasks - n_running_tasks; //n. tasks to collect
#ifdef LOG
      cerr << "* to collect: " << n_to_collect << endl;
#endif
      //sleeping_time = (last_scheduling_time / n_active_tasks) / P_WAITING_APPROX; //set the sleeping time
      //last_scheduling_time = 0;
      void *data = NULL;
      while(true) {
	//pick a collected task
	if(farm.load_result(&data)) {
	  
	  //dispatch
	  out_task_t *ack = (out_task_t *) data;
	  if(!ack->ack) {
	    //sample: send out
#ifdef DATASIZE
	    datasize += sizeof(out_task_t) + (size_t)n_monitors * sizeof(multiplicityType);
	    if(ixtime < 0)
	      ixtime = get_xtime_from(0, xt);
	    else
	      thr = (double)datasize/get_xtime_from(ixtime, xt);
#endif
	    ff_send_out(data);
	  }
	  else {
	    //ack: rescheduling
	    simulation_task_t *task = tasks[ack->simulation_task_id];
#ifdef LOG
	    cerr << "collected ack: sim. #" << ack->simulation_id << " (task #" << ack->simulation_task_id << ")"
		 << " (" << task->n_sampled 
		 << " on " << n_samples << ")" << endl;
#endif
	    bars.set(ack->simulation_task_id, 100 * ack->n_sampled / n_samples);
	    //last_scheduling_time += ack->running_time; //update running time for the scheduling step
	    if(task->n_sampled == n_samples) {
	      //end of the task
	      n_active_tasks--;
#ifdef LOG
	      cerr << "end of the sim. #" << ack->simulation_id << " (task #" << ack->simulation_task_id << ")" << endl;
#endif
	    }
	    else {
	      ready_queue.push(task); //put in RQ
	    }
	    ack->~out_task_t();
	    FREE(ack);
	    n_to_collect--;
	    if(n_to_collect == 0) break;
	  }
	}
	//usleep(max((useconds_t)sleeping_time, (useconds_t)WAITING_THRESHOLD)); //suspend the waiting thread
      }
    }
    /*
      scheduling loop ends here
    */
    
    //send EOS and join
    farm.offload((void *)FF_EOS);
    farm.wait();
    delete collector;
    /*
      ----------------------------------
      SIM. FARM ends here
      ----------------------------------
    */

#else //defined USE_FF_ACCEL
    /*
      seq. version begins here
    */
    ProgressBars bars(n_simulations);
    for(int i=0; i<n_simulations; ++i) {
      simulation_task_t * t = tasks[i];
#ifdef LOG
      ofstream &logfile(t->simulation->get_logfile());
#endif
      int n_sampled = 0;
      double next_sample_time = n_sampled * sampling_period; //first uncovered sample
      double tau = t->simulation->restore(); //restore the last tau computed
      double next_reaction_time = t->simulation->get_time() + tau;
      bool stall = t->simulation->get_stall();

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

#ifdef LOG
	  logfile << "*** sampling at time " << next_sample_time << endl;
#endif
	  sample_t *monitor = t->simulation->monitor();

#ifdef DATASIZE
	  datasize += sizeof(out_task_t) + (size_t)n_monitors * sizeof(multiplicityType);
	  if(ixtime < 0) {
	    ixtime = get_xtime_from(0, xt);
	  }
	  else
	    thr = (double)datasize/get_xtime_from(ixtime, xt);
#endif

	  //send output
	  void *p = MALLOC(sizeof(out_task_t));
	  ff_send_out((void *)new (p) out_task_t(t, next_sample_time, monitor)); //send out the sample
	  ++n_sampled;
	  if(!(n_sampled % (n_samples/100)))
	    bars.set(i, 100 * n_sampled / n_samples);
	  //check termination
	  if(n_sampled == n_samples) {
#ifdef LOG
	    logfile << "--- END OF THE TASK ---\n";
#endif
	    running_time = get_time_from(start_time, usage);
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
      } //end sim. loop
      t->running_time = running_time;
    }
    //*** end sampling engine
    /*
      seq. version ends here
    */
#endif //defined USE_FF_ACCEL
    
    cout << endl;

#ifdef TIME
    for(int i=0; i<n_simulations; ++i)
      running_times[i] = tasks[i]->running_time;
#endif
#ifdef GRAIN
    for(int i=0; i<n_simulations; ++i) {
      cycle_ticks[i] = tasks[i]->simulation->get_cycle_ticks();
      steps[i] = tasks[i]->simulation->get_steps();
    }
#endif

    //clean-up
    for(int i=0; i<n_simulations; ++i) {
      tasks[i]->~simulation_task_t();
      FREE(tasks[i]);
    }

    tasks.~vector<simulation_task_t *>();
    FREE(&tasks);

    return ((void *)FF_EOS);
  }

#ifdef DATASIZE
  void print_datasize_stats() {
    cout << "datasize: " << (double)datasize/1048576 << " MBytes (" << datasize << " Bytes)"
	 << ", throughput: " << 1000*thr/1048576 << " MBytes/s"
	 << endl;
  }
#endif

#ifdef GRAIN
  void print_grain_stats() {
    sort(cycle_ticks.begin(), cycle_ticks.end());
    double m = 0, avg_steps = 0;
    for(int i=0; i<n_simulations; ++i) {
      m += cycle_ticks[i];
      avg_steps += steps[i];
    }
    m /= n_simulations;
    avg_steps /= n_simulations;
    double v = 0;
    for(int i=0; i<n_simulations; ++i) {
      double v_ = cycle_ticks[i] - m;
      v += v_ * v_;
    }
    v /= (n_simulations - 1);
    cout << "min: " << cycle_ticks[0] << " ticks"
	 << ", max: " << cycle_ticks[n_simulations-1] << " ticks"
	 << ", avg: " << m << " ticks"
	 << ", sd: " << sqrt(v) << " ticks"
	 << ", n. steps (avg): " << avg_steps
	 << endl;
  }
#endif

#ifdef TIME
  void print_time_stats() {
    sort(running_times.begin(), running_times.end());
    double m = 0;
    for(int i=0; i<n_simulations; ++i)
      m += running_times[i];
    m /= n_simulations;
    double v = 0;
    for(int i=0; i<n_simulations; ++i) {
      double v_ = running_times[i] - m;
      v += v_ * v_;
    }
    v /= (n_simulations - 1);
    cout << "min: " << running_times[0]/1000 << " s"
	 << ", max: " << running_times[n_simulations-1]/1000 << " s"
	 << ", avg: " << m/1000 << " s"
	 << ", sd: " << sqrt(v) << " ms"
	 << endl;
  }
#endif

private:
  int n_simulations;
  unsigned int n_monitors;
  double sampling_period;
  double time_limit;
  int n_samples;
#ifdef USE_FF_ACCEL
  int n_workers;
  int samples_per_slice;
  int p_inflight;
#endif
#ifndef USE_FF_ACCEL
  struct rusage usage;
#endif
#ifdef DATASIZE
  size_t datasize;
  double thr;
  struct timeval xt;
  double ixtime;
#endif
#ifdef TIME
  vector<double> running_times;
#endif
#ifdef GRAIN
  vector<double> cycle_ticks;
  vector<unsigned int> steps;
#endif
  int rank;
};





/*
  ------------------------------------------------------------------------------
  TrajectoriesAlignment:
  - send out samples
  ------------------------------------------------------------------------------
*/
class TrajectoriesAlignment: public ff_node {
public:
  TrajectoriesAlignment(const cwc_parameters_t &param, const unsigned int n_monitors, int n_samples, int rank = -1) :
    n_simulations(param.n_simulations),
    n_monitors(n_monitors),
    n_samples(n_samples),
    raw_output(param.raw_output),
    stats_enabled(param.stats_enabled),
    rank(rank)
  {
    //sampling queues
    for(unsigned int i=0; i<n_simulations; ++i) {
      sq.push_back(new uSWSR_Ptr_Buffer(n_samples / 10));
      sq[i]->init();
    }
    tq = new uSWSR_Ptr_Buffer(n_samples / 10);
    tq->init();
    newest_sample_time = -1;

    //raw output
    if(raw_output) {
      for(unsigned int i=0; i<n_simulations; i++) {
	stringstream outfile_name;
	outfile_name << param.outfile_prefix << "_s" << i;
	ofstream *outfile = new ofstream(outfile_name.str().c_str());
	if(!outfile->is_open()) {
	  cerr << "Could not open file: " << outfile_name.str() << endl;
	  exit(1);
	}
	raw_files.push_back(outfile);
      }
    }
    sizes = new vector<unsigned long>(n_simulations,0);

    push_counter = 0;
#ifdef USE_FF_ACCEL
    bulk_size = n_simulations * n_samples / param.n_slices;
#else
    bulk_size = n_samples * n_simulations; //all data
#endif

    //time and throughput misuration
#if defined(DATASIZE) || defined(TIME)
    ixtime = -1;
#endif
#ifdef DATASIZE
    max_buffer_len = 0;
    datasize = 0;
#endif
#ifdef TIME
    cxtime = get_xtime_from(0, xt);
    svc_time = 0;
    svc_first_time = -1;
#endif
  }

  int svc_init() {
    if(rank >= 0) {
      ff_mapThreadToCpu(rank);
      cerr << "TrajectoriesAlignment pinned to core #" << rank << endl;
    }
    return 0;
  }

  void svc_end() {
    //clean-up
    for(unsigned int i=0; i<n_simulations; ++i)
      if(sq[i])
	delete sq[i];
    if(tq)
      delete tq;
    if(sizes)
      delete sizes;
    if(raw_output)
      for(unsigned int i=0; i<n_simulations; ++i) {
	raw_files[i]->close();
	delete raw_files[i];
      }
  }

  void* svc(void *task) {
#ifdef TIME      
    double rtime_ = get_xtime_from(0, xt);
    if(svc_first_time < 0)
      svc_first_time = rtime_;
#endif
    out_task_t *sample_task = (out_task_t *) task;
    int simulation_id = sample_task->simulation_id;
    double sample_time = sample_task->sample_time;
    sample_t *sim_sample = sample_task->monitor;
    FREE(sample_task);
    push_counter = (push_counter + 1) % bulk_size;

    //buffer sample
    sq[simulation_id]->push(sim_sample);

    if(sample_time > newest_sample_time) {
      double *ptime = (double *)MALLOC(sizeof(double));
      *ptime = sample_time;
      tq->push(ptime);
      newest_sample_time = sample_time;
    }
    ++sizes->at(simulation_id);
 
#ifdef DATASIZE
    size_t max_buffer_len_ = 0;
    for(unsigned int i=0; i<n_simulations; ++i)
      max_buffer_len_ = std::max(max_buffer_len_, sizes->at(i));
    max_buffer_len = std::max(max_buffer_len_, max_buffer_len);
#endif

    //raw output
    if(raw_output)
      write_sample_to_file(*raw_files[simulation_id], sample_time, *sim_sample);

    if(!push_counter) {
      long unsigned int n_pop = (long unsigned int)n_samples;
      for (unsigned int i=0; i< n_simulations; ++i)
	n_pop = (std::min)(n_pop, sizes->at(i));
      while(n_pop--) {
#if defined(TIME) || defined(DATASIZE)
	if(ixtime < 0)
	  ixtime = get_xtime_from(0, xt);
#endif
#ifdef DATASIZE
	datasize += sizeof(window_task_t) + (size_t)n_simulations * (size_t)n_monitors * sizeof(multiplicityType);
	wt = get_xtime_from(ixtime, xt);
	thr = (double)datasize/wt;
#endif
	if(stats_enabled) {
	  //send out window-task
	  sample_t **instant_samples = (sample_t **)MALLOC(n_simulations * sizeof(sample_t *)); //to be deallocated
	  double *sample_time_;
	  tq->pop((void **)&sample_time_);
	  for(unsigned int j=0; j<n_simulations; j++) {
	    sq[j]->pop((void **)&instant_samples[j]);
	    sizes->at(j)--;
	  }
	  void *p = MALLOC(sizeof(window_task_t)); //to be deallocated
	  ff_send_out(new (p) window_task_t(*sample_time_, instant_samples));
	  FREE(sample_time_);
	}
	else {
	  //destroy samples
	  sample_t *s;
	  double *sample_time_;
	  tq->pop((void **)&sample_time_);
	  for(unsigned int j=0; j<n_simulations; j++) {
	    sq[j]->pop((void **)&s);
	    sizes->at(j)--;
	    delete s;
	  }
	  FREE(sample_time_);
	}
      }
    }
#ifdef TIME
    svc_time += get_xtime_from(rtime_, xt);
    wt = get_xtime_from(ixtime, xt);
#endif
    return GO_ON;
  }

#ifdef DATASIZE
  void print_datasize_stats() {
    cout << "max. buffer size: " << max_buffer_len << endl;
    cout << "datasize: " << (double)datasize/1048576 << " MBytes (" << datasize << " Bytes)"
	 << ", throughput: " << 1000*thr/1048576 << " MBytes/s"
	 << endl;
  }
#endif

#ifdef TIME
  void print_time_stats() {
    cout << "-\nTrajectoriesAlignment svc time: " << svc_time/1000 << " s" << endl
	 << "creation-to-work latency: " << (ixtime - cxtime)/1000 << " s" << endl
	 << "svc-to-work latency: " << (ixtime - svc_first_time)/1000 << " s" << endl
	 << "working time: " << wt/1000 << " s" << endl;
  }
#endif

private:
  unsigned int n_simulations;
  unsigned int n_monitors;
  int n_samples;
  //sampling queues:
  // - one for each sim.
  // - queue-vector is indexed by sampling instants
  // - queue items: samples (one per monitor) at instant t
  vector<uSWSR_Ptr_Buffer *> sq;
  uSWSR_Ptr_Buffer *tq;
  double newest_sample_time;
  bool raw_output;
  vector<ofstream *> raw_files;
  bool stats_enabled;
#if defined(DATASIZE) || defined(TIME)
  double wt, ixtime;
  struct timeval xt;
#endif
#ifdef DATASIZE
  size_t max_buffer_len;
  size_t datasize;
  double thr;
#endif
#ifdef TIME
  double svc_time, svc_first_time, cxtime;
#endif
  vector<unsigned long> *sizes;
  unsigned int push_counter, bulk_size;
  int rank;
};





/*
  ------------------------------------------------------------------------------
  WindowsGenerator:
  - generate stat. windows
  INPUT: sample_t ** (array of samples - one per simulation - at instant t)
  OUTPUT: Stat_Window * (stat. window cenetered at t)
  ------------------------------------------------------------------------------
*/
class WindowsGenerator: public ff_node {
public:
  WindowsGenerator(cwc_parameters_t &param, unsigned int n_monitors, int rank=-1) :
    n_simulations(param.n_simulations),
    n_monitors(n_monitors),
    time_limit(param.time_limit),
    rank(rank)
  {

    //stat windows
    for(unsigned int i=0; i<n_monitors; ++i)
      stat_windows.push_back(new Stat_Window<multiplicityType>(param.window_size, n_simulations));

#if defined(DATASIZE) || defined(TIME)
    ixtime = -1;
    cxtime = get_xtime_from(0, xt);
#endif
#ifdef DATASIZE
    datasize = 0;
#endif
#ifdef TIME
    svc_time = 0;
#endif
  }

  int svc_init() {
    if(rank >= 0) {
      ff_mapThreadToCpu(rank);
      cerr << "WindowsGenerator pinned to core #" << rank << endl;
    }
#ifdef USE_FF_DISTR
    cout << "windows progress: 0%\r" << flush;
#endif
    return 0;
  }

  void svc_end() {
#ifdef USE_FF_DISTR
    cout << "\rwindows progress: 100%" << endl;
#endif
    //clean-up
    for(int i=0; i<n_monitors; ++i)
      delete stat_windows[i];
  }

  void * svc(void * task) {
#ifdef TIME
    double rtime_ = get_xtime_from(0, xt);
#endif
    window_task_t *t = (window_task_t *)task;
    sample_t **instant_samples = t->samples;
    double sample_time = t->sample_time;
    t->~window_task_t();
    FREE(t);
    
    for(int i=0; i<n_monitors; i++)
      stat_windows[i]->slide(sample_time);

    //fill front of monitor-window
    for(int j=0; j<n_simulations; j++) {
      for(int i=0; i<n_monitors; i++)
	stat_windows[i]->add_front(instant_samples[j]->at(i), j);
      delete instant_samples[j];
    }
    FREE(instant_samples);

    //print progress
#ifdef USE_FF_DISTR
    double percent = (sample_time / time_limit * 100);
    cout << "windows progress: " << (int)percent << "%\r" << flush;
#endif

    //send copies of stat. windows
    //TODO: MALLOC
    vector<Stat_Window<multiplicityType> *> *out = new vector<Stat_Window<multiplicityType> *>;
    for(int i=0; i<n_monitors; ++i) {
      out->push_back(new Stat_Window<multiplicityType>(*stat_windows[i]));

      //time and throughput misurations start here
#ifdef DATASIZE
      datasize += sizeof(Stat_Window<multiplicityType>) + out->at(i)->get_sizeof();
#endif
    }
#if defined(TIME) || defined(DATASIZE)
    if(ixtime < 0)
      ixtime = get_xtime_from(0, xt);
#endif
#ifdef DATASIZE
    thr = (double)datasize/get_xtime_from(ixtime, xt);
#endif
#ifdef TIME
    svc_time += get_xtime_from(rtime_, xt);
#endif
    //time and throughput misurations end here

    return out;
  }

#ifdef DATASIZE
  void print_datasize_stats() {
    cout << "datasize: " << (double)datasize/1048576 << " MBytes (" << datasize << " Bytes)"
	 << ", throughput: " << 1000*thr/1048576 << " MBytes/s"
	 << endl;
  }
#endif

#ifdef TIME
  void print_time_stats() {
    cout << "WindowsGenerator svc time: " << svc_time/1000 << " s" << endl;
    cout << "creation-to-work latency: " << (ixtime - cxtime)/1000 << " s" << endl;
  }
#endif

private:
  int n_simulations;
  int n_monitors;
  double time_limit;
	
  //statistical windows (one for each monitor)
  vector<Stat_Window<multiplicityType> *> stat_windows;

#if defined(TIME) || defined(DATASIZE)
  struct timeval xt;
  double ixtime, cxtime;
#endif
#ifdef DATASIZE
  size_t datasize;
  double thr;
#endif
#ifdef TIME
  double svc_time;
#endif
  int rank;
};





/*
  ------------------------------------------------------------------------------
  Stat. Engine:
  - compute/write stats
  INPUT: Stat_Window<multiplicityType> * (array of stat. windows - one per monitor)
  ------------------------------------------------------------------------------
*/
class StatEngine: public ff_node {
public:
  StatEngine(cwc_parameters_t &param, const unsigned int n_monitors, const string &model_label, vector<string> &labels, int rank=-1)
    :
    n_monitors(n_monitors),
    sampling_period(param.sampling_period),
    time_limit(param.time_limit),
#ifdef USE_STAT_ACCEL
    n_stat_workers(param.n_stat_workers),
#endif
    rank(rank)
  {
    stat_windows = NULL;
    
    //set statistical engines: one toolset for each monitor
    stat_engines = vector<vector<Statistical_Engine<multiplicityType> *> >(n_monitors);
    for(unsigned int i=0; i<n_monitors; i++) {
      stat_engines[i].push_back(new MeanVariance_Engine<multiplicityType>(param.grubb/100, ENABLE_GNUPLOT));
      stat_engines[i].push_back(new Quantiles_Engine<multiplicityType>(param.n_quantiles, ENABLE_GNUPLOT));
      stat_engines[i].push_back(new Kmeans_Engine<multiplicityType>(param.n_clusters, param.window_size, ENABLE_GNUPLOT, sampling_period, param.degree, param.prediction_factor));
      //stat_engines[i].push_back(new QT_Engine<multiplicityType>(param.qt_threshold, param.window_size, ENABLE_GNUPLOT, sampling_period, param.degree, param.prediction_factor));
      stat_engines[i].push_back(new Peaks_Engine<multiplicityType>(param.window_size, param.peaks_parameter1, param.peaks_parameter2, param.n_simulations, ENABLE_GNUPLOT, sampling_period, param.degree, param.prediction_factor));
      stringstream monitor_id;
      monitor_id << i;
      for(unsigned int j=0; j<stat_engines[i].size(); j++) {
	stat_engines[i][j]->init(
				 param.outfile_prefix,
				 "monitor" + monitor_id.str(),
				 model_label,
				 labels[i],
				 param.n_simulations,
				 time_limit,
				 sampling_period * param.prediction_factor
				 );
      }
      /*
      //gnuplot-all for kmeans
      if(outfiles)
      ((Kmeans_Engine<multiplicityType> *)stat_engines[i][2])->gnuplot_all(outfile_prefix, "monitor" + monitor_id.str(), time_limit, i, n_simulations);
      */
      
    }
    n_engines = stat_engines[0].size(); //assuming n_monitors > 0 for safety

#ifdef USE_STAT_ACCEL
    St = new ff_farm<>(true); //set stat. farm-acc.
    lb = new my_loadbalancer(n_stat_workers);
    Se = new StatFarm_Emitter(n_stat_workers, n_monitors, lb);
    for(int i=0; i<n_stat_workers; ++i)
      stat_workers.push_back(new StatFarm_Worker(&stat_engines));
    St->add_emitter(Se);
    St->add_workers(stat_workers);
#endif

#ifdef TIME
    svc_time = -1;
#endif
  }

  int svc_init() {
    if(rank >= 0) {
      ff_mapThreadToCpu(rank);
      cerr << "StatEngine pinned to core #" << rank << endl;
    }
    return 0;
  }

  void svc_end() {
    for(int i=0; i<n_monitors; ++i) {
      //finalize statistics
      for(unsigned int j=0; j<stat_engines[i].size(); ++j)
	stat_engines[i][j]->finalize(*stat_windows->at(i));
      //free stat. windows
      stat_windows->at(i)->free_all();
      delete stat_windows->at(i);
      //delete statistical engines
      for(unsigned j=0; j<stat_engines[i].size(); j++)
	delete stat_engines[i][j];
    }
    delete stat_windows;
      
#ifdef USE_STAT_ACCEL
    delete St;
    delete Se;
    delete lb;
#endif

#ifdef TIME
    svc_time = get_xtime_from(svc_time, xt);
#endif
  }

  void * svc(void * task) {
#ifdef TIME
    if(svc_time < 0)
      svc_time = get_xtime_from(0, xt);
#endif
    if(stat_windows) {
      //delete stat. windows
      for(int i=0; i<n_monitors; ++i) {
	stat_windows->at(i)->free_oldest();
	delete stat_windows->at(i);
      }
      delete stat_windows;
    }
    stat_windows = (vector<Stat_Window<multiplicityType> *> *)task;

#ifdef USE_STAT_ACCEL
    //farm-accelerated statistics (by monitors)
    St->run_then_freeze();
    for(int i=0; i<n_monitors; ++i)
      for(int j=0; j<n_engines; ++j) {
	void *p = MALLOC(sizeof(stat_task_t));
	stat_task_t *out = new (p) stat_task_t(stat_windows->at(i), i, j);
	St->offload(out);
      }
    St->offload((void *)FF_EOS);
    St->wait_freezing();
#else
    //sequential statistics
    for(int i=0; i<n_monitors; ++i)
      for(unsigned int j=0; j<stat_engines[i].size(); ++j)
	stat_engines[i][j]->compute_write(*stat_windows->at(i));
#endif
    return GO_ON;
  }

#ifdef TIME
  void print_time_stats() {
    vector<double> ta(stat_engines[0].size(), 0);
    double total = 0;
    for(int i=0; i<n_monitors; ++i)
      for(unsigned int j=0; j<ta.size(); ++j) {
	double t = stat_engines[i][j]->get_running_time();
	ta[j] += t;
	total += t;
      }
    cout << "-\nStat. Stage TIME stats\n"
	 << "svc time: " << svc_time/1000 << " s" << endl
	 << "stat. engines total running time: " << total/1000 << " s" << endl;
    for(unsigned int i=0; i<ta.size(); ++i)
      cout << "stat. engine #" << i << ": " << (ta[i] / n_monitors) / 1000 << " s" << endl;
  }
#endif
  
private:
  int n_monitors;
  int n_engines;
  double sampling_period;
  double time_limit;
	
  //statistical engines
  vector<vector<Statistical_Engine<multiplicityType> *> > stat_engines;

  //the most recent stat. windows
  vector<Stat_Window<multiplicityType> *> *stat_windows;

#ifdef USE_STAT_ACCEL
  int n_stat_workers;
  ff_farm<> *St;
  my_loadbalancer *lb;
  StatFarm_Emitter *Se;
  vector<ff_node *> stat_workers;
#endif
  int rank;
#ifdef TIME
  double svc_time;
  struct timeval xt;
#endif
};
#endif
