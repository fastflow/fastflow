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

#ifndef FFACCEL_H
#define FFACCEL_H
#include <ctime>
#include <sstream>
#include <deque>
#include "definitions.h"
#include "Simulation.h"
#include "statistics.h"
#include "kmeans.h"
#include <ff/farm.hpp>
#include <ff/node.hpp>
using namespace ff;
using namespace std;

#ifdef FF_ALLOCATOR
#include <ff/allocator.hpp>
#define ALLOCATOR_INIT()
#define MALLOC(size)   (FFAllocator::instance()->malloc(size))
#define FREE(ptr) (FFAllocator::instance()->free(ptr))
#else
#define ALLOCATOR_INIT() 
#define MALLOC(size)   malloc(size)
#define FREE(ptr) free(ptr)
#endif

struct simulation_task_t {
  //input task
  simulation_task_t(Simulation *sim, double sp):
    simulation(sim),
    last_sample_time(-sp),
    stop_time(0),
    running_time(0) {}

  Simulation *simulation;
  double last_sample_time;
  double stop_time;
  double running_time;
};

class simulation_task_comparison {
public:
  bool operator() (simulation_task_t * &lhs, simulation_task_t * &rhs) {
    //speed comparison (the lower task has higher priority)
    return (lhs->stop_time / lhs->running_time) > (rhs->stop_time / rhs->running_time);
  }
};

struct out_task_t {
  out_task_t(simulation_task_t *st, double t, sample_t *m):
    ack(false),
    simulation_task(st),
    sample_time(t),
    running_time(-1),
    monitor(m) {}

  out_task_t(simulation_task_t *st, double t):
    ack(true),
    simulation_task(st),
    sample_time(-1),
    running_time(t),
    monitor(NULL) {}

  ~out_task_t() {
    //simulation_task->~simulation_task_t();
    //FREE(simulation_task);
    if(monitor) {
      monitor->~sample_t();
      free(monitor);
    }
  }

  bool ack;
  simulation_task_t *simulation_task;
  double sample_time;
  double running_time;
  sample_t *monitor;
};


class Worker: public ff_node {
public:
  Worker(double t) :
    sampling_period(t) {}
  
  //int svc_init() { return 0;}
  void * svc(void * task) {
    simulation_task_t * t = (simulation_task_t *)task;
    //cout << "(Worker) start\n";
    double current_time = t->simulation->get_time();
    double last_sample_time = t->last_sample_time; //last covered sample
    double next_sample_time = last_sample_time + sampling_period; //first uncovered sample

    //| start timer ...
    //getrusage(RUSAGE_SELF, &start_usage_stats); //per-process
    
    gettimeofday(&time_misuration, NULL);
    double start_time = (double)time_misuration.tv_sec * 1000000.0 + (double)time_misuration.tv_usec;

    while(true) {
      if(next_sample_time <= current_time || current_time == -1) {
	//fresh state : monitor()
	sample_t *monitor = t->simulation->monitor();
	//send output
	void *p = MALLOC(sizeof(out_task_t));
	ff_send_out((void *)new (p) out_task_t(t, next_sample_time, monitor));
	//update the task
	last_sample_time = next_sample_time;
	next_sample_time += sampling_period;
	if(next_sample_time > t->stop_time) {

	  //... stop timer |
	  gettimeofday(&time_misuration, NULL);
	  double stop_time = (double)time_misuration.tv_sec * 1000000.0 + (double)time_misuration.tv_usec;

	  t->last_sample_time = last_sample_time;
	  //cout << "break condition: " << next_sample_time << " > " << t->stop_time << endl;
	  p = MALLOC(sizeof(out_task_t));
	  return ((void *)new (p) out_task_t(t, stop_time - start_time));
	}
      }
      else
	//stale state : step()
	current_time = t->simulation->step();
    }
    return GO_ON; //never reached
  }

private:
  double sampling_period;
  struct timeval time_misuration;
  //struct rusage start_usage_stats;
  //struct rusage stop_usage_stats;
};


class Collector: public ff_node {
public:
  //Collector(int ns, int nm, vector<ofstream *> *rwf, ofstream *rdf, vector<Reduction *> &r, int preallocation = 1) :
  Collector(int ns, int nm, vector<ofstream *> *rwf, ofstream *rdf, ofstream *cf, ofstream *clf, double g, vector<int> qi, int nc, int preallocation = 1) :
    n_simulations(ns),
    n_monitors(nm),
    raw_files(rwf),
    reduced_file(rdf),
    classifications_file(cf),
    clusters_file(clf),
    grubb(double(1) - g / 100),
    //significance(cp / 100),
    front_samples(0),
    q_indexes(qi),
    n_quantiles(qi.size()),
    n_clusters(nc)
  {
    //sample windows (integer and real)
    m = (multiplicityType **)MALLOC(n_monitors * sizeof(multiplicityType *));
    m_double = (double **)MALLOC(n_monitors * sizeof(double *));
    for(int i=0; i<n_monitors; i++) {
      m[i] = (multiplicityType *)MALLOC(n_simulations * sizeof(multiplicityType));
      m_double[i] = (double *)MALLOC(n_simulations * sizeof(double));
    }
    //sampling buffers
    for(int j=0; j<n_simulations; j++)
      samples.push_back(new deque<sample_t *>(/*preallocation*/));
    //clustering datarow
    clstr_datarow = (double **)MALLOC(n_simulations * sizeof(double *));
    for(int j=0; j<n_simulations; j++)
      clstr_datarow[j] = (double *)MALLOC(1 * sizeof(double));
    //clustering centers and sizes
    last_centers = (double **)MALLOC(n_monitors * sizeof(double *));
    for(int i=0; i<n_monitors; i++) {
      last_centers[i] = (double *)MALLOC(n_clusters * sizeof(double));
      for(int j=0; j<n_clusters; j++)
	last_centers[i][j] = double(0);
    }
    last_sizes = (int **)MALLOC(n_monitors * sizeof(int *));
    for(int i=0; i<n_monitors; i++)
      last_sizes[i] = (int *)MALLOC(n_clusters * sizeof(int));
    first_pass = true;
  }

  ~Collector() {
    for(int i=0; i<n_monitors; i++) {
      FREE(m[i]);
      FREE(m_double[i]);
    }
    FREE(m);
    FREE(m_double);
    for(int i=0; i<n_simulations; i++) {
      delete samples[i];
      FREE(clstr_datarow[i]);
    }
    FREE(clstr_datarow);
    //clustering centers and sizes
    for(int i=0; i<n_monitors; i++) {
      FREE(last_centers[i]);
    }
    FREE(last_centers);
    for(int i=0; i<n_monitors; i++) {
      FREE(last_sizes[i]);
    }
    FREE(last_sizes);
  }

  void * svc(void * task) {
    out_task_t *out = (out_task_t *) task;
    if(out->ack) return out; //ACK

    int simulation_id = out->simulation_task->simulation->get_id();
    double time = out->sample_time;
    sample_t *sample = out->monitor;
#ifdef LOG
    cerr << "time " << time << "; # " << simulation_id << ": ";
    for(int j=0; j<n_monitors; j++) cerr << sample->at(j) << " ";
    cerr << endl;
#endif
    FREE(out);
    //buffer data
    if(samples[simulation_id]->empty())
      front_samples++;
    samples[simulation_id]->push_back(sample);

    //raw output
    if(raw_files)
      write_sample_to_file(*raw_files->at(simulation_id), time, *sample); //write

    //reduce (if instant-sampling is complete)
    if(front_samples == n_simulations) {
      //build M[i][j]: i-th monitor for the j-th simulation
      for(int j=0; j<n_simulations; j++) {
	sample = samples[j]->front();
	samples[j]->pop_front(); //remove from the queue
	for(int i=0; i<n_monitors; i++)
	  m[i][j] = sample->at(i);
	delete sample;
      }

      //build sample with real numbers
      for(int i=0; i<n_monitors; i++) {
	sort(m[i], m[i] + n_simulations); //sort
	for(int j=0; j<n_simulations; j++)
	  m_double[i][j] = m[i][j];
      }
      //now samples are ordered



      //compute statistics
      //double m_r[n_monitors][2]; //reductions
      //multiplicityType m_c[q_indexes.size()][n_monitors]; //classifications
      //int clustering[n_simulations]; //clustering membership
      double** m_r = new double *[n_monitors];
      for(int i=0; i<n_monitors; i++)
	m_r[i] = new double[2];
      multiplicityType **m_c = new multiplicityType *[n_quantiles];
      for(int i=0; i<n_quantiles; i++)
	m_c[i] = new multiplicityType[n_monitors];
      int *clustering = new int[n_simulations];
      double **centers = NULL; //clustering centers

      for(int i=0; i<n_monitors; i++) {
	//compute reductions
	//1. mean and variance (under Grubb's test)
	grubb_reductions(m[i], n_simulations, grubb, m_r[i][0], m_r[i][1]);

	//compute classifications
	//1. quantiles
	//multiplicityType q[n_quantiles];
	multiplicityType *q = new multiplicityType[n_quantiles];
	quantiles(m[i], q_indexes, q);
	//put pins in classifications
	for(int j=0; j<n_quantiles; j++)
	  m_c[j][i] = q[j];
	delete[] q;

	//compute clustering
	//1. k-means
	for(int j=0; j<n_simulations; j++)
	  clstr_datarow[j][0] = m[i][j];
	if(first_pass) {
	  //initialize centers
	  vector<int> init = linspace_int(n_simulations, n_clusters);
	  for(int j=0; j<n_clusters; j++)
	    last_centers[i][j] = clstr_datarow[init[i]][j];
	  first_pass = false;
	}
	centers = seq_kmeans(clstr_datarow, 1, n_simulations, n_clusters, 0.1, clustering, last_sizes[i], last_centers[i]);
	for(int j=0; j<n_clusters; j++) {
	  last_centers[i][j] = centers[j][0];
	}
      }
	
	
	
      //write reductions
      stringstream reductions;
      reductions << time << "\t";
      for(int i=0; i<n_monitors; i++)
	//1. mean and variance (under Grubb's test)
	reductions << m_r[i][0] << "\t" << m_r[i][1] << "\t";
      string out_r = reductions.str();
      out_r.erase(out_r.length() - 1); //erase the last \t
      out_r += "\n";
      write_to_file(*reduced_file, out_r);
      //cleanup
      for(int i=0; i<n_monitors; i++)
	delete[] m_r[i];
      delete[] m_r;

      //write classifications
      stringstream classifications;
      classifications << time << "\t";
      for(int i=0; i<n_monitors; i++) {
	//1. quantiles
	for(int j=0; j<n_quantiles; j++)
	  classifications << m_c[j][i] << "\t";
      }
      string out_c = classifications.str();
      out_c.erase(out_c.length() - 1); //erase the last \t
      out_c += "\n";
      write_to_file(*classifications_file, out_c);
      //cleanup
      for(int i=0; i<n_quantiles; i++)
	delete[] m_c[i];
      delete[] m_c;

      //write clusters
      stringstream clstr;
      clstr << time << "\t";
      for(int i=0; i<n_monitors; i++) {
	//1. k-means
	for(int j=0; j<n_clusters; j++)
	  clstr << last_centers[i][j] << "\t" << last_sizes[i][j] << "\t";
      }
      string out_clstr = clstr.str();
      out_clstr.erase(out_clstr.length() - 1); //erase the last \t
      out_clstr += "\n";
      write_to_file(*clusters_file, out_clstr);
      //cleanup
      delete[] clustering;
      free(centers[0]);
      free(centers);



      //update queues
      front_samples = 0;
      for(int j=0; j<n_simulations; j++)
	if(!samples[j]->empty())
	  front_samples++;
      
    }
    
    return GO_ON;
  }

private:
  int n_simulations;
  int n_monitors;
  vector<ofstream * > *raw_files;
  ofstream *reduced_file;
  ofstream *classifications_file;
  ofstream *clusters_file;
  vector<deque<sample_t * > * > samples; //sampling buffers
  multiplicityType **m; //sample window (integer)
  double **m_double; //sample window (real)
  double **clstr_datarow;
  double **last_centers; //last clustering centers
  int **last_sizes;
  bool first_pass;
  double grubb;
  int front_samples;
  vector<int> q_indexes;
  int n_quantiles;
  int n_clusters;
};
#endif
