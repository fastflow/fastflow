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

#include "ff/platforms/platform.h"
//#define BOOST_PROGRAM_OPTIONS_NO_LIB 1
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <cmath>
#include <ctime>
//#include <sys/time.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <algorithm>
using namespace std;

#include "Driver.h"
#include "random.h"
#include "Simulation.h"
#include "output.h"
#include "ProgressBars.h"

#ifdef HYBRID
#include "ode.h"
#define RATE_CUTOFF_DEFAULT RATE_CUTOFF_INFINITE //to be changed
#define POPULATION_CUTOFF_DEFAULT POPULATION_CUTOFF_INFINITE //to be changed
#endif

#ifdef USE_FF_ACCEL
#include "ff_accel.hpp"
#include "statistics.h"
//#include <unistd.h> //usleep
//statistics
#define GRUBB_DEFAULT 50
#define N_QUANTILES_DEFAULT 5
#define N_CLUSTERS_DEFAULT 2
//parallelism
#define N_WORKERS_DEFAULT 2
#define N_SLICES_DEFAULT 10
//gnuplot
#define GNUPLOT_TERMINAL "postscript enhanced eps 20"
#define GNUPLOT_EXTENSION "eps"
#define N_PLOTTED_QUANTILES 3
#endif

#ifdef LOCKFREE
#define P_INFLIGHT_DEFAULT 50 //must be < 100
#define P_WAITING_APPROX 13
#define WAITING_THRESHOLD 10000 //microseconds
#endif

#define FIXED_SEED 536478



int main(int argc, char *argv[]) {
  double time_limit;
  int n_simulations;
#ifdef HYBRID
  double rate_cutoff;
  multiplicityType population_cutoff;
#endif
#ifdef USE_FF_ACCEL
  double grubb;
  int n_quantiles, n_clusters;
  int n_workers, n_slices;
#ifdef LOCKFREE
  int p_inflight;
#endif
#endif





  /* --- Command-line options --- */

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "this help message")
    ("input-file,i", po::value<string>(), "input file")
    ("output-file,o", po::value<string>(), "output file")
    ("time-limit,t", po::value<double>(&time_limit)->default_value(0), "time limit")
    ("sampling-period,s", po::value<double>(), "sampling period")
    ("simulations,n", po::value<int>(&n_simulations)->default_value(1), "number of simulations")
#ifdef HYBRID
    //hybrid
    ("rate-cutoff,r", po::value<double>(&rate_cutoff)->default_value(RATE_CUTOFF_DEFAULT), "minimum rate cutoff")
    ("population-cutoff,p", po::value<multiplicityType>(&population_cutoff)->default_value(POPULATION_CUTOFF_DEFAULT), "minimum population cutoff")
#endif
#ifdef USE_FF_ACCEL
    //statistics
    ("grubb,g", po::value<double>(&grubb)->default_value(GRUBB_DEFAULT), "%-level for Grubb's test")
    ("quantiles,q", po::value<int>(&n_quantiles)->default_value(N_QUANTILES_DEFAULT), "number of quantiles (> 2)")
    ("clusters,c", po::value<int>(&n_clusters)->default_value(N_CLUSTERS_DEFAULT), "number of clusters")
    ("raw-output", "raw output")
    //parallelism
    ("workers,w", po::value<int>(&n_workers)->default_value(N_WORKERS_DEFAULT), "number of workers")
    ("slices", po::value<int>(&n_slices)->default_value(N_SLICES_DEFAULT), "number of time-limit fractions for scheduling")
#ifdef LOCKFREE
    ("inflight", po::value<int>(&p_inflight)->default_value(P_INFLIGHT_DEFAULT), "% of inflight tasks")
#endif
#endif
    ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if(vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  //input file
  ifstream *infile = NULL;
  if(vm.count("input-file")) {
    infile = new ifstream(vm["input-file"].as<string>().c_str());
    if(!infile->good()) {
      delete infile;
      cerr << "Could not open file: " << vm["input-file"].as<string>() << endl;
      return 1;
    }
  } else {
    cerr << "No input file provided" << endl;
    return 1;
  }

  //output file(s)
  vector<ofstream *> *outfiles;
  string out_fname = vm["output-file"].as<string>();
  bool other_output = false;
#ifdef USE_FF_ACCEL
  other_output = true;
#endif
  if(vm.count("raw-output") || !other_output) {
    outfiles = new vector<ofstream *>;
    if(vm.count("output-file")) {
      for(int i=0; i<n_simulations; i++) {
	stringstream outfile_name;
	outfile_name << out_fname << "_s" << i;
	ofstream *outfile = new ofstream(outfile_name.str().c_str());
	if(!outfile->is_open()) {
	  cerr << "Could not open file: " << outfile_name.str() << endl;
	  return 1;
	}
	outfiles->push_back(outfile);
      }

    } else {
      cerr << "No output file provided" << endl;
      return 1;
    }
  }
  else outfiles = NULL;

  //sampling period
  double sampling_period = -1;
  if(vm.count("sampling-period"))
    sampling_period = vm["sampling-period"].as<double>();
  else {
    cerr << "No sampling period provided" << endl;
    return 1;
  }
  if(sampling_period <= 0) {
    cerr << "Invalid sampling period: " << sampling_period << endl;
    return 1;
  }

  //number of quantiles
  if (n_quantiles < 3) {
    n_quantiles = N_QUANTILES_DEFAULT;
    cerr << "number of quantiles ajusted to " << N_QUANTILES_DEFAULT << endl;
  }

  //number of clusters
  if (n_clusters < 1) {
    n_clusters = N_CLUSTERS_DEFAULT;
    cerr << "number of clusters ajusted to " << N_CLUSTERS_DEFAULT << endl;
  }





  //parse
  scwc::Driver driver;
  bool result = driver.parse_stream(*infile, vm["input-file"].as<string>().c_str());
  if(result) {
    cout << "Simulating: " << driver.model->title << " (" << n_simulations << " runs):" << endl;
#ifdef HYBRID
    cout << "Hybrid (stochastic/deterministic) semantics" << endl;
    if(rate_cutoff != RATE_CUTOFF_INFINITE)
      cout << "Minimum rate cutoff: " << rate_cutoff << endl;
    else
      cout << "Minimum rate cutoff: " << "infinite" << endl;
    if(population_cutoff != POPULATION_CUTOFF_INFINITE)
      cout << "Minimum population cutoff: " << population_cutoff << endl;
    else
      cout << "Minimum population cutoff: " << "infinite" << endl;
#else
    cout << "Pure stochastic semantics" << endl;
#endif

    //adjust time values
    int n_samples = (int)ceil(time_limit / sampling_period);
    cout << "Number of samples: " << n_samples << endl;
    time_limit = n_samples * sampling_period;
    cout << "Time-limit: " << time_limit << endl;

    //seed generator
    rng_type seed_rng((unsigned int)time(0));
    //rng_type seed_rng(FIXED_SEED);
    uint_gm_type seed_gm(0, (numeric_limits<int>::max)());
    uint_vg_type seed_vg(seed_rng, seed_gm);

    //simulations train
    vector<Simulation *> simulations;
    cerr << "seeds:" << endl;
    for(int i=0; i<n_simulations; i++) {
      //get a generator
      int sim_seed = seed_vg();
      cerr << sim_seed << endl;
      rng_type sim_rng(sim_seed);
      u01_gm_type sim_gm;
      u01_vg_type sim_vg(sim_rng, sim_gm);
      simulations.push_back(new Simulation(i, *driver.model, sim_vg
#ifdef HYBRID
					   , rate_cutoff, population_cutoff
#endif
					   ));
      if(outfiles)
	write_to_file(*outfiles->at(i), driver.model->header(time_limit)); //header
    }

#ifdef SIMD
    cout << "SIMD instruction-set (SSE) enabled" << endl;
    simd = "simd_";
#endif

    //prepare for time-misuration
    struct timeval time_misuration;
    stringstream time_fname;
    time_fname << out_fname << "_time";
    string simd = "";
#ifndef USE_FF_ACCEL    
    time_fname << "_sequential" << "_" << simd << n_simulations << "x";
#else
    time_fname << "_parallel" << "_" << simd << n_simulations << "x" << "_" << n_workers << "w";
#endif
    ofstream time_file(time_fname.str().c_str());









    /* ----- PARALLEL variants ----- */
#ifdef USE_FF_ACCEL
    int n_monitors = driver.model->monitors.size(); //number of monitors

    stringstream gnuplot_base;
    gnuplot_base << "set grid" << endl;
    gnuplot_base << "set xlabel \"time\"" << endl;
    gnuplot_base << "set terminal " << GNUPLOT_TERMINAL << endl;

    //file for reduced output
    ofstream *reduced_file = new ofstream((out_fname + "_reductions").c_str());
    if(!reduced_file->is_open()) {
      cerr << "Could not open file: " << out_fname + "_reductions" << endl;
      return 1;
    }
    //header of the reductions
    stringstream header;
    header << "# Reductions for the model: " << driver.model->title << endl;
    header << "#" << endl;
    header << "# Reductions:" << endl;
    header << "# Sample mean and variance (under " << grubb << "%-level Grubb's test)" << endl;
    header << "#" << endl;
    header << "# Monitors:" << endl;
    for(int i=0; i<n_monitors; i++)
      header << "# (" << i << ") " << driver.model->monitors[i]->title << endl;
    header << "#" << endl;
    header << "# mean (0)\tvariance(0)";
    for(int i=1; i<n_monitors; i++)
      header << "\tmean (" << i << ")\tvariance(" << i << ")";
    header << endl;
    write_to_file(*reduced_file, header.str());
    //gnuplot scripts
    for(int i=0; i<n_monitors; i++) {
      stringstream r_gnuplot;
      r_gnuplot << gnuplot_base.str();
      r_gnuplot << "set ylabel \"mean and standard deviation (" << driver.model->monitors[i]->title << ")\"" << endl;
      //file
      stringstream fns;
      fns << out_fname << "_mean_sd_monitor" << i << ".gnu";
      ofstream f(fns.str().c_str());
      if(!f.is_open()) {
	cerr << "Could not open file: " << fns << endl;
	return 1;
      }
      //script
      stringstream s;
      s << r_gnuplot.str();
      s << "set output \"" << out_fname << "_mean_sd_monitor" << i << "." << GNUPLOT_EXTENSION << "\"" << endl;
      //standard deviation
      s << "plot \"" << out_fname << "_reductions\" using 1:" << 2 + (2 * i) << ":(sqrt($" << 2 + (2 * i) + 1 << "))";
      s << " title \"standard deviation\""<< " with yerrorbars lt 2,\\" << endl;
      //mean
      s << "\"" << out_fname << "_reductions\" using 1:" << 2 + (2 * i);
      s << " title \"mean (under " << grubb << "%-grubb Grubb's test)\""<< " with lines lw 2 lt 1" << endl;
      write_to_file(f, s.str());
      f.close();
    }

    //classifications
    //file
    ofstream *classifications_file = new ofstream((out_fname + "_classifications").c_str());
    if(!classifications_file->is_open()) {
      cerr << "Could not open file: " << out_fname + "_classifications" << endl;
      return 1;
    }
    //compute quantile indexes
    vector<int> q_indexes = linspace_int(n_simulations, n_quantiles);
    //header
    stringstream c_header;
    c_header << "# Classifications for the model: " << driver.model->title << endl;
    c_header << "#" << endl;
    c_header << "# Classifications:" << endl;
    //quantiles guidelines
    c_header << "# Quantiles: ";
    for(int i=0; i<n_quantiles; i++)
      c_header << (float(q_indexes[i])/(n_simulations - 1)) * 100 << "%, ";
    //c_header << float(q_indexes[n_quantiles-1])/(n_simulations - 1) * 100 << "%" << endl;
    c_header << "#" << endl;
    c_header << "# Monitors:" << endl;
    for(unsigned int i=0; i<driver.model->monitors.size(); i++) c_header << "# (" << i << ") " << driver.model->monitors[i]->title << endl;
    c_header << "#" << endl;
    //columns
    c_header << "# quantile 0% (0)";
    for(int j=1; j<n_quantiles; j++)
      c_header << "\tquantile " << (float(q_indexes[j])/(n_simulations - 1)) * 100 << "% (0)";
    for(int i=1; i<n_monitors; i++)
      for(int j=0; j<n_quantiles; j++)
	c_header << "\tquantile " << (float(q_indexes[j])/(n_simulations - 1)) * 100 << "% (" << i << ")";
    c_header << endl;
    write_to_file(*classifications_file, c_header.str());
    //gnuplot scripts
    for(int i=0; i<n_monitors; i++) {
      stringstream c_gnuplot;
      c_gnuplot << gnuplot_base.str();
      c_gnuplot << "set ylabel \"quantiles (" << driver.model->monitors[i]->title << ")\"" << endl;
      //file
      stringstream fns;
      fns << out_fname << "_quantiles_monitor" << i << ".gnu";
      ofstream f(fns.str().c_str());
      if(!f.is_open()) {
	cerr << "Could not open file: " << fns << endl;
	return 1;
      }
      //script
      stringstream s;
      s << c_gnuplot.str();
      s << "set output \"" << out_fname << "_quantiles_monitor" << i << "." << GNUPLOT_EXTENSION << "\"" << endl;
      //first non-zero quantile in selected range
      float q_level = (float(q_indexes[1])/(n_simulations - 1)) * 100;
      s << "plot \"" << out_fname << "_classifications\" using 1:" << 2 + (n_quantiles * i) + 1 << " title \"" << q_level << "% quantile\" with lines";
      //all remaining quantiles but 100%
      for (int j=2; j<n_quantiles-1; j++) {
	float q_level = (float(q_indexes[j])/(n_simulations - 1)) * 100;
	s << ",\\" << endl << "\"" << out_fname << "_classifications\" using 1:" << 2 + (n_quantiles * i) + j << " title \"" << q_level << "% quantile\"with lines";
      }
      s << endl;
      write_to_file(f, s.str());
      f.close();
    }

    //file for clustering output
    ofstream *clusters_file = new ofstream((out_fname + "_clusters").c_str());
    if(!reduced_file->is_open()) {
      cerr << "Could not open file: " << out_fname + "_clusters" << endl;
      return 1;
    }
    //header of the clusters
    stringstream header_clstr;
    header_clstr << "# Clusters for the model: " << driver.model->title << endl;
    header_clstr << "#" << endl;
    header_clstr << "# K-means clustering (with sizes)" << endl;
    header_clstr << "#" << endl;
    header_clstr << "# Monitors:" << endl;
    for(int i=0; i<n_monitors; i++) header_clstr << "# (" << i << ") " << driver.model->monitors[i]->title << endl;
    header_clstr << "#" << endl;
    //columns
    header_clstr << "# cluster 1 (0)";
    for(int j=1; j<n_clusters; j++)
      header_clstr << "\tcluster " << j+1 << " (0)";
    for(int i=1; i<n_monitors; i++)
      for(int j=0; j<n_clusters; j++)
	header_clstr << "\tcluster " << j+1 << " (" << i << ")";
    header_clstr << endl;
    write_to_file(*clusters_file, header_clstr.str());
    //gnuplot scripts
    for(int i=0; i<n_monitors; i++) {
      stringstream r_gnuplot;
      r_gnuplot << gnuplot_base.str();
      r_gnuplot << "set ylabel \"clusters (" << driver.model->monitors[i]->title << ")\"" << endl;
      //file
      stringstream fns;
      fns << out_fname << "_clusters_monitor" << i << ".gnu";
      ofstream f(fns.str().c_str());
      if(!f.is_open()) {
	cerr << "Could not open file: " << fns << endl;
	return 1;
      }
      //script
      stringstream s;
      s << r_gnuplot.str();
      s << "set output \"" << out_fname << "_clusters_monitor" << i << "." << GNUPLOT_EXTENSION << "\"" << endl;
      //centers
      s << "plot \"" << out_fname << "_clusters\" using 1:" << 2 + 2 * n_clusters * i;
      s << " title \"cluster 1\""<< " with lines";
      for(int j=1, ci=2; j<n_clusters; j++, ci+=2) {
	s << ",\\" << endl;
	s << "\"" << out_fname << "_clusters\" using 1:" << 2 + ci + 2 * n_clusters * i;
	s << " title \"cluster " << j + 1 << "\""<< " with lines";
      }
      s << endl;
      write_to_file(f, s.str());
      f.close();
    }


    
    //tasks train
    vector<simulation_task_t *> tasks;
    for(int i=0; i<n_simulations; i++) {
      void *p = MALLOC(sizeof(simulation_task_t));
      simulation_task_t * t = new (p) simulation_task_t(simulations[i], sampling_period);
      tasks.push_back(t);
    }

    //set the farm
    ff_farm<> farm(true, n_simulations); //use the number of simulations as queue-length
    vector<ff_node *> workers;
    for(int i=0; i<n_workers; i++)
      workers.push_back(new Worker(sampling_period));
    farm.add_workers(workers);

    //set execution-time misuration
    gettimeofday(&time_misuration, NULL);
    double start_misuration = (double)time_misuration.tv_sec + (double)time_misuration.tv_usec / 1000000.0;

    //print headers
    cout << "Parallel simulations" << endl;
    cout << "Number of workers: " << n_workers << endl;
    cout << "Level for Grubb's test: " << grubb << "%" << endl;
    cout << "Number of quantiles: " << n_quantiles << endl;
    cout << "Number of clusters: " << n_clusters << endl;
    





    /* --- 1. locking scheduling --- */
#ifndef LOCKFREE
    cout << "Starting farm with locking scheduling" << endl;

    //set slicing
    int samples_per_slice =  n_samples / n_slices;
    int last_slice_size = n_samples % samples_per_slice;
    int total_slices = (last_slice_size == 0)? n_slices : n_slices + 1;
    double slicing_period = time_limit / n_slices;
    //cout << "last slice size: " << last_slice_size << endl;
    //cout << "n. slices (full): " << n_slices << endl;
    //cout << "total slices: " << total_slices << endl;
    //cout << "slicing period: " << slicing_period << endl;

    cout << "scheduling step: time-limit / " << n_slices << " = " << slicing_period << ")" << endl;
    //start the progress-bars
    ProgressBars bars(n_simulations);

    Collector *collector = new Collector(n_simulations, n_monitors, outfiles, reduced_file, classifications_file, clusters_file, grubb, q_indexes, n_clusters, samples_per_slice);
    farm.add_collector(collector);

    //full-window slices
    for(int i=0; i<n_slices; i++) { //scheduling loop

      //make the farm accept tasks
      farm.run_then_freeze();

      //offload all the tasks in the train
      double stop_time;
      for(int j=0; j<n_simulations; j++) {
	simulation_task_t *t = tasks[j];
	t->stop_time += slicing_period;
	stop_time = t->stop_time;
	farm.offload(t); //offload
      }
      //cout << "offloaded simulations till " << stop_time << endl;

      farm.offload((void *)FF_EOS); //locking EOS

      //join (to unlock)
      farm.wait_freezing();
      //progress_bar((i + 1) * 100 / total_slices); //update the progress-bar
      for(int j=0; j<n_simulations; j++)
	bars.set(j, (i + 1) * 100 / total_slices);
    }

    //last trunked-window slice
    if(last_slice_size > 0) {
      slicing_period = (slicing_period / samples_per_slice) * last_slice_size;
      //make the farm accept tasks
      farm.run_then_freeze();

      //offload all the tasks in the train
      double stop_time;
      for(int j=0; j<n_simulations; j++) {
	simulation_task_t *t = tasks[j];
	t->stop_time += slicing_period;
	stop_time = t->stop_time;
	farm.offload(t); //offload
      }
      //cout << "(extra) offloaded simulations till " << stop_time << endl;

      farm.offload((void *)FF_EOS); //locking EOS

      //join (to unlock)
      farm.wait_freezing();
      for(int j=0; j<n_simulations; j++)
	bars.set(j, 100);
    }





    /* --- 2. non-locking scheduling --- */
#else
    cout << "Starting farm with lock-free scheduling (";

    double slicing_period = time_limit / n_slices;
    int samples_per_slice = n_samples / n_slices;

    cout << "scheduling step: time-limit / " << n_slices << " = " << slicing_period;
    cout << "; " << p_inflight << "% inflight)" << endl;

    ProgressBars bars(n_simulations);

    int n_active_tasks = n_simulations;
    int n_running_tasks = 0;
    priority_queue<simulation_task_t *, vector<simulation_task_t *>, simulation_task_comparison> ready_queue;

    Collector *collector = new Collector(n_simulations, n_monitors, outfiles, reduced_file, classifications_file, clusters_file, grubb, q_indexes, n_clusters, samples_per_slice);
    farm.add_collector(collector);

    //fill the ready-queue
    for(int i=0; i<n_simulations; i++) {
      tasks[i]->stop_time = slicing_period;
      ready_queue.push(tasks[i]);
    }

    //make the farm accept tasks
    farm.run_then_freeze();

    //set sleeping for the active-waiting thread
    double sleeping_time = 0, last_scheduling_time = 0;

    while(true) { //scheduling loop

      if(n_active_tasks > n_workers) {
	//we need scheduling
#ifdef LOG
	cerr << "scheduling step" << endl;
#endif
	//offload RQ
	while(!ready_queue.empty()) {
	  simulation_task_t *t = ready_queue.top();
	  ready_queue.pop();
	  farm.offload(t); //offload
#ifdef LOG
	  cerr << "offloaded simulation #" << t->simulation->get_id();
	  cerr << " (speed: " << (t->stop_time / t->running_time) << ")";
	  cerr << " till " << t->stop_time << /*" on " << time_limit <<*/ endl;
#endif
	}
	//(active) waiting for some tasks
	n_running_tasks = max(p_inflight * n_active_tasks / 100, 1);
	int n_to_collect = n_active_tasks - n_running_tasks; //tasks to collect
	sleeping_time = (last_scheduling_time / n_active_tasks) / P_WAITING_APPROX; //set the sleeping time
	last_scheduling_time = 0;
	void *data = NULL;
	while(true) {
	  //pick a collected task
	  if(farm.load_result(&data)) {
	    out_task_t *ack = (out_task_t *) data;
	    simulation_task_t *task = ack->simulation_task;
#ifdef LOG
	    cerr << "collected " << ack->simulation_task->simulation->get_id();
	    cerr << " (" << ack->simulation_task->stop_time << ")" << endl;
#endif
	    bars.set(ack->simulation_task->simulation->get_id(), 100 * ack->simulation_task->stop_time / time_limit);
	    last_scheduling_time += ack->running_time; //update running time for the scheduling step
	    if(task->stop_time >= time_limit) {
	      //end of the task
	      //cout << " (reached time-limit)" << endl;
	      task->~simulation_task_t();
	      FREE(task);
	      n_active_tasks--;
	    }
	    else {
	      //cout << endl;
	      task->running_time += ack->running_time; //update running time for the task
	      task->stop_time = min(task->stop_time + slicing_period, time_limit);
	      ready_queue.push(task); //put in RQ
	    }
	    ack->~out_task_t();
	    FREE(ack);
	    n_to_collect--;
	    if(n_to_collect == 0) break;
	  }
	  usleep(max((useconds_t)sleeping_time, (useconds_t)WAITING_THRESHOLD)); //suspend the waiting thread
	}
      }

      else {
	//we don't need scheduling
	//cout << "\nwe don't need scheduling\n";
	if(n_running_tasks > 0) {
	  //offload RQ
	  while(!ready_queue.empty()) {
	    simulation_task_t *t = ready_queue.top();
	    ready_queue.pop();
	    farm.offload(t); //offload
	    //cout << "offloaded simulation #" << t->simulation->get_id();
	    //cout << " (speed: " << (t->stop_time / t->running_time) << ")";
	    //cout << " till " << t->stop_time << /*" on " << time_limit <<*/ endl;
	  }
	  //(active) waiting for all the tasks
	  sleeping_time = (last_scheduling_time / n_active_tasks) / P_WAITING_APPROX; //set the sleeping time
	  while(true) {
	    void *data = NULL;
	    //pick a collected task
	    if(farm.load_result(&data)) {
	      out_task_t *ack = (out_task_t *) data;
	      simulation_task_t *task = ack->simulation_task;
	      bars.set(task->simulation->get_id(), 100 * task->stop_time / time_limit);
	      if(task->stop_time >= time_limit) {
		//end of the task
		task->~simulation_task_t();
		FREE(task);
	      }
	      else {
		task->running_time += ack->running_time; //update running time for the task
		task->stop_time = time_limit; //will run till the end
		ready_queue.push(task); //put in RQ
	      }
	      FREE(ack);
	      n_active_tasks--;
	      if(n_active_tasks == 0) break;
	    }
	    usleep(max((useconds_t)sleeping_time, (useconds_t)WAITING_THRESHOLD)); //suspend the waiting thread
	  }
	}
	break;

      }
    }

    //here: all active tasks are in RQ
    //cout << "here: all active tasks are in RQ" << endl;
    //set time_limit as stop_time for all active tasks, then offload
    for(int i=0; i<n_simulations; i++)
      if(tasks[i]) tasks[i]->stop_time = time_limit;
    //offload RQ
    while(!ready_queue.empty()) {
      simulation_task_t *t = ready_queue.top();
      ready_queue.pop();
      farm.offload(t); //offload
      //cout << "offloaded simulation #" << t->simulation->get_id();
      //cout << " (speed: " << (t->stop_time / t->running_time) << ")";
      //cout << " till " << t->stop_time << /*" on " << time_limit <<*/ endl;
    }
    //send EOS and join
    farm.offload((void *)FF_EOS);
    farm.wait_freezing();
    for(int i=0; i<n_simulations; i++)
      bars.set(i, 100);
    //progress_bar(100);
#endif

    //cout << "done" << endl;

#ifdef TRACE_FASTFLOW
    farm.ffStats(cerr);
#endif

    //clean-up
    delete collector;
    for(int i=0; i<n_simulations; i++) {
      simulation_task_t *t = tasks[i];
      FREE(t);
    }
    reduced_file->close();
    classifications_file->close();
    clusters_file->close();
    delete reduced_file;
    delete classifications_file;
    delete clusters_file;








    /* ----- SEQUENTIAL version ----- */
#else
    //start timer
    gettimeofday(&time_misuration, NULL);
    double start_misuration = (double)time_misuration.tv_sec + (double)time_misuration.tv_usec / 1000000.0;

    cout << "Sequential simulations" << endl;
    //start the progress-bars
    ProgressBars bars(n_simulations);

    for(int i=0; i<n_simulations; i++) {
      //cout << "running simulation #" << i << endl;
      
      Simulation *simulation = simulations[i];
      ofstream &outfile = *outfiles->at(i);
      double current_time = 0;
      double next_sample_time = 0; //first uncovered sample

      while(true) {
	if(next_sample_time <= current_time || current_time == -1) {
	  //fresh state : monitor()
	  bars.set(i, (next_sample_time/time_limit) * 100);
	  vector<multiplicityType> *monitor = simulation->monitor();
	  write_sample_to_file(outfile, next_sample_time, *monitor);
	  delete monitor;
	  next_sample_time += sampling_period;
	  if(next_sample_time > time_limit) {
	    //cout << "break condition: " << next_sample_time << " > " << time_limit << endl;
	    break;
	  }
	}
	else
	  //stale state : step()
	  current_time = simulation->step();
      }

      //progress_bar((i + 1) * 100 / n_simulations);
      bars.set(i, 100);
      
    }
#endif





    /* --- close and clean-up --- */

    //stop timer and write misuration
    gettimeofday(&time_misuration, NULL);
    double stop_misuration = (double)time_misuration.tv_sec + (double)time_misuration.tv_usec / 1000000.0;
    stringstream time_row;
    time_row << (stop_misuration - start_misuration);
    write_to_file(time_file, time_row.str());
    time_file.close();

    //simulations clean-up
    for(int i=0; i<n_simulations; i++) {
      delete simulations[i];
    }
  }

  //files clean-up
  if(outfiles) {
    for(unsigned int i=0; i<outfiles->size(); i++) {
      outfiles->at(i)->close();
      delete outfiles->at(i);
    }
    delete outfiles;
  }
  delete infile;
  cout << endl;
  return 0;
}
