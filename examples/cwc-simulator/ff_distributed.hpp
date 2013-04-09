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

#ifndef _CWC_FF_DISTRIBUTED_HPP_
#define _CWC_FF_DISTRIBUTED_HPP_

/*
 *
 *   -------------------------------------------------------------------
 *  |                                                 host0             |
 *  |                                          ----------------------   |  
 *  |                                         |                      |  |
 *  |                                         |                      |  |
 *  |          master                   ------|-> Worker --> Sender -|--
 *  |    --------------------------    |      |                      |
 *  |   |                          |   |      |    (ff_pipeline)     |          
 *  v   |                          |   |       ----------------------
 *   ---|-> Collector    Emitter ->|---            C           P 
 *  ^   |                          |   | SCATTER (COMM1)
 *  |   |      (ff_pipeline)       |   | 
 *  |    --------------------------    |              host1
 *  |          C            P          |       ----------------------
 *  | FROM ANY (COMM2)                 |      |                      |
 *  |                                  |      |                      |
 *  |                                   ------|-> Worker --> Sender -|--
 *  |                                         |                      |  |
 *  |                                         |    (ff_pipeline)     |  |
 *  |                                          ----------------------   |
 *  |                                               C          P        |
 *   -------------------------------------------------------------------    
 *
 *   COMM1, the server address is master:port1 (the address1 parameter)
 *   COMM2, the server address is master:port2 (the address2 parameter)
 *
 */

#include <ff/svector.hpp>
#include <ff/dnode.hpp>
#include <ff/pipeline.hpp>
#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>
using namespace ff;

#include <queue>
using namespace std;

#include <parameters.hpp>
#include <random.h>
#include <Driver.h>
#include <Simulation.h>
#include <ff_definitions.hpp>
#ifdef USE_FF_ACCEL
#include <ff_accel.hpp>
#endif
#include <ProgressBars.hpp>

#define COMM1 zmqScatter
#define COMM2 zmqFromAny   
// gloabals just to save some coding
// the following SPSC unbounded queue is shared between the Worker and the Sender threads
// to free unused memory
uSWSR_Ptr_Buffer RECYCLE(1024); // NOTE: See also the farm_farm test for another solution to the same problem !



/*
  ------------------------------------------------------------------------------
  SeedsProducer:
  - produce and scatter seeds
  ------------------------------------------------------------------------------
*/
class SeedsProducer: public ff_dnode<COMM1> {
  typedef COMM1::TransportImpl  transport_t;
protected:
  static void callback(void * e,void *arg) {
#ifdef LOG
    cerr << "emitter callback on " << e << ", " << arg << "\n";
#endif
    int *array = (int *) e;
    delete [] array;
  }
public:
  SeedsProducer(cwc_parameters_t &param, const string &name, transport_t* const transp) :
    nTasks(param.n_simulations),
    nHosts(param.nhosts),
    fixed_seed(param.fixed_seed),
    fixed_seed_value(param.fixed_seed_value),
    name(name),
    address(param.scatter_address),
    transp(transp)
  {
    //sim_per_site = nTasks/nHosts;
    int acc = 0;
    sim_per_site = new int[nHosts];
    ifstream pf("part");
    unsigned int i;
    for(i=0; i<nHosts-1; ++i) {
      float part;
      pf >> part;
      sim_per_site[i] = (int)((part / 100) * nTasks);
      acc += sim_per_site[i];
    }
    sim_per_site[i] = nTasks - acc;
  }
    
  int svc_init() {
    // the callback will be called as soon as the output message is no 
    // longer in use by the transport layer
#ifdef LOG
    cerr << "init " << name << ", producer channel @ " << address << endl;
#endif
    ff_dnode<COMM1>::init("scatter", address, nHosts, transp, true /* prod */, 0, callback); 
#ifdef LOG
    cerr << "init SCATTER completed (producer)\n";
#endif
    return 0;
  }
  void svc_end() {
    delete[] sim_per_site;
#ifdef LOG
    cerr << "Emitter: DONE\n";
#endif
  }
    
  void * svc(void* task) {
#ifdef LOG
    cerr << name << " scattering seeds:\n";
#endif

    //seed generator
    rng_b_type seed_rng(fixed_seed ? fixed_seed_value : (unsigned int)time(0));
    uint_gm_type seed_gm(0, (numeric_limits<int>::max)());
    uint_vg_type seed_vg(seed_rng, seed_gm);
    int **seeds = new int*[nHosts];
    int offset = 0;
    
    for(unsigned i=0;i<nHosts;++i) {
      seeds[i] = new int[1+sim_per_site[i]];
      seeds[i][0] = offset;
      for(int j=0; j<sim_per_site[i]; ++j, ++offset)
	seeds[i][j+1] = seed_vg();
#ifdef LOG
      for(int j=0; j<sim_per_site[i]; ++j)
	cerr << seeds[i][j] << " ";
      cerr << endl;
#endif
    }
    //return GO_ON;
    ff_send_out(seeds);
    return (void *)FF_EOS;
  }

  //prepare the task for the 0mq channel
  // prepare is called nHosts times, one time for each destination
  void prepare(svector<iovec>& v, void* ptr, const int dest) {
#ifdef LOG
    cerr << name << ": preparing data for scattering seeds to " << dest << " ptr  " << ptr << "\n";
#endif
    int * num = new int(1 + sim_per_site[dest]); 
    int **seeds = (int **) ptr; //difficult to handle delete; 
    struct iovec tasknum={num, sizeof(int)};
    struct iovec iov={seeds[dest], *num * sizeof(int)};
    v.push_back(tasknum);
    v.push_back(iov);
  }

private:
  int *sim_per_site;
  unsigned int nTasks;
  unsigned int nHosts;
  bool fixed_seed;
  int fixed_seed_value;
  //unsigned int sim_per_site;
protected:
  const std::string name;
  const std::string address;
  transport_t   * transp;
};





/*
  ------------------------------------------------------------------------------
  TrajectoriesProducer:
  - send out samples
  INPUT: out_task_t * (sample)
  OUTPUT: vector<out_task_t *> * (bulk of samples)
  ------------------------------------------------------------------------------
*/
class TrajectoriesProducer: public ff_dnode<COMM2> {
  typedef COMM2::TransportImpl transport_t;

protected:
  static void callback(void *e, void*a) {
    vector<void *> *v = (vector<void *> *)a;
    for(unsigned int i=0; i<v->size(); ++i) {
      out_task_t* p = (out_task_t *) v->at(i);
      p->~out_task_t();
    }
  }

public:
  TrajectoriesProducer(const cwc_parameters_t &param, const unsigned int n_monitors, const string &name, transport_t* const transp) :
    nHosts(param.nhosts),
    n_monitors(n_monitors),
    name(name),
    address(param.fromany_address),
    transp(transp)
  {
    msg_buffer = new vector<void *>;
  }

  int svc_init() {
    // the callback will be called as soon as the output message is no 
    // longer in use by the transport layer
    //cout << "init " << name << ", producer channel @ " << address << endl;
    ff_dnode<COMM2>::init("gather", address, nHosts, transp, true/*prod*/, transp->getProcId(), /*callback*/ NULL);  
#ifdef LOG
    cerr << " FromAny init done (producer)\n";
#endif
    return 0;
  }
  void svc_end() {
#ifdef LOG
    cerr << "Sender: DONE\n";
#endif
  }

  void* svc(void* task) {
    out_task_t *sample_task = (out_task_t *) task;
#ifdef LOG
    cerr << "buffering sample @ " << sample_task->sample_time
	 << " for sim. # " << sample_task->simulation_id << endl;
#endif
    msg_buffer->push_back(sample_task);
    if(msg_buffer->size() == msg_buffer_size) {
#ifdef LOG
      cerr << "send out" << endl;
#endif
      ff_send_out(msg_buffer);
      msg_buffer = new vector<void *>;
    }
    return GO_ON;
  }

  void set_buffer_size(unsigned int s) {
    msg_buffer_size = s;
  }

  void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
    vector<void *> *buf = (vector<void *> *)ptr;
    for(unsigned int i=0; i<buf->size(); ++i) {
      struct iovec iov1 = {buf->at(i), sizeof(out_task_t)};
      v.push_back(iov1);
      out_task_t * ot = (out_task_t *) buf->at(i);
      struct iovec iov2 = {&ot->monitor->at(0), n_monitors * sizeof(multiplicityType)};
      v.push_back(iov2);
      RECYCLE.push(ot->monitor); // put it in the delete queue
    }
    setCallbackArg(ptr);
  }

private:
  const unsigned nHosts;
  vector<void *> *msg_buffer;
  unsigned int msg_buffer_size;
  unsigned int n_monitors;

protected:
  const std::string name;
  const std::string address;
  transport_t * transp;
};





/*
  ------------------------------------------------------------------------------
  SeedsConsumer:
  - consume seeds
  - parse input model (and build model-objects)
  - prepare sim. tasks
  ------------------------------------------------------------------------------
*/
class SeedsConsumer: public ff_dnode<COMM1> {
  typedef COMM1::TransportImpl transport_t;
public:
  SeedsConsumer(const cwc_parameters_t &param, TrajectoriesProducer *tp, const string& name, transport_t* const transp) :
    nHosts(param.nhosts),
    fname(param.infname),
    tp(tp),
    name(name),
    address(param.scatter_address),
    transp(transp) {}
    
  int svc_init() {
#ifdef LOG
    cerr << "init " << name << ", consumer channel @ " << address << " nodeID " << transp->getProcId() << endl;
#endif
    ff_dnode<COMM1>::init("scatter", address, nHosts, transp, false /*cons*/, transp->getProcId());
#ifdef LOG
    cerr << "Worker: init SCATTER completed (consumer)\n";
#endif
    return 0;
  }

  void svc_end() {
#ifdef LOG
    cerr << "Worker: DONE\n";
#endif
  }

  void * svc(void * task) {
#ifdef LOG
    cerr << "Worker: received " << n_simulations << " seeds\n";
#endif

    //parse
    scwc::Driver driver;
    ifstream *infile = new ifstream(fname.c_str());
    if(!infile->good()) {
      delete infile;
      cerr << "Could not open file: " << fname << endl;
      exit(1);
    }
    if(!(driver.parse_stream(*infile, fname.c_str()))) {
      //syntax error
      cerr << fname << ": syntax error." << endl;
      exit(1); //ff::error?
    }
#ifdef LOG
    cerr << "syntax ok" << endl;
#endif

    //prepare sim. tasks and send them out
    int *sv = (int *)task;
    int offset = sv[0]; //sim. id offset
    void *p = MALLOC(sizeof(vector<simulation_task_t *>(n_simulations)));
    vector<simulation_task_t *> *tasks = new (p) vector<simulation_task_t *>(n_simulations);
    for(int i=0; i<n_simulations; i++) {
      // create a simulation
      int sim_seed = sv[i]; // get from a vector of seeds
      Simulation *fsp;
      MYNEW(fsp, Simulation, offset+i, *driver.model, sim_seed
#ifdef HYBRID
	    , rate_cutoff
	    , population_cutoff
	    , sampling_period
#endif
	    );
      p = MALLOC(sizeof(simulation_task_t));
      tasks->at(i) = new (p) simulation_task_t(fsp, offset);
    }

#ifdef LOG
    cerr << "sending out sim. tasks" << endl;
#endif
    tp->set_buffer_size(n_simulations); //to be changed
    ff_send_out(tasks);
    return NULL;
  }

  virtual void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
    int* n =static_cast<int*>(v[0]->operator[](0)->getData());
    n_simulations = *n - 1;
    int *t=static_cast<int*>(v[0]->operator[](1)->getData());
    task=t;
  }

private:
  unsigned nHosts;
  const string fname;
  int n_simulations;
  TrajectoriesProducer *tp;

protected:
  const std::string name;
  const std::string address;
  transport_t * transp;
};





/*
  ------------------------------------------------------------------------------
  TrajectoriesConsumer:
  - gather samples
  - produce aligned trajectories
  ------------------------------------------------------------------------------
*/
class TrajectoriesConsumer: public ff_dnode<COMM2> {
  typedef COMM2::TransportImpl transport_t;

public:    
  TrajectoriesConsumer(const cwc_parameters_t &param, const unsigned int n_monitors, const int n_samples, const string& name, transport_t* const transp) :
    n_simulations(param.n_simulations),
    nHosts(param.nhosts),
    n_monitors(n_monitors),
    raw_output(param.raw_output),
    stats_enabled(param.stats_enabled),
    name(name),
    address(param.fromany_address),
    transp(transp)
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

#ifdef DATASIZE
    sizes = new vector<unsigned long>(n_simulations,0);
    max_buffer_len = 0;
    datasize = 0;
    itime = -1;
    gettimeofday(&time_misuration, NULL);
    ctime = (double)time_misuration.tv_sec + (double)time_misuration.tv_usec / 1000000.0;
#endif

    push_counter = 0;
#ifdef USE_FF_ACCEL
    bulk_size = n_simulations * n_samples / param.n_slices;
#else
    bulk_size = n_samples * n_simulations; //all data
#endif
  }
    
  // initializes dnode
  int svc_init() {
    //cout << "init " << name << ", consumer channel @ " << address << endl;
    ff_dnode<COMM2>::init("gather", address, nHosts, transp, false /* cons */, 0);
#ifdef LOG
    cerr << "FromAny init done (consumer)\n";
#endif
    return 0;
  }
  
  void svc_end() {
    //clean-up
    for(unsigned int i=0; i<n_simulations; ++i)
      if(sq[i])
	delete sq[i];
    if(tq)
      delete tq;
#ifdef DATASIZE
    if(sizes)
      delete sizes;
#endif
    if(raw_output)
      for(unsigned int i=0; i<n_simulations; ++i) {
	raw_files[i]->close();
	delete raw_files[i];
      }
#ifdef LOG
    cerr << "Collector: DONE\n";
#endif
  }
   
  void *svc(void *task) {
    vector<out_task_t *> *v = (vector<out_task_t *> *)task;
#ifdef LOG
    cerr << "got " << v->size() << " samples" << endl;
#endif

    for(unsigned int i=0; i<v->size(); ++i) {
      //got result: consume
      out_task_t *sample_task = v->at(i);
      bool frontier = true;
      int simulation_id = sample_task->simulation_id;
      double sample_time = sample_task->sample_time;
      sample_t *sim_sample = sample_task->monitor;
      delete sample_task->msg;
      push_counter = (push_counter + 1) % bulk_size;

      //buffer sample
      sq[simulation_id]->push(sim_sample);

      if(sample_time > newest_sample_time) {
	tq->push(new double(sample_time));
	newest_sample_time = sample_time;
      }

#if defined(DATASIZE)
      ++sizes->at(simulation_id);
      size_t max_buffer_len_ = 0;
      for(unsigned int i=0; i<n_simulations; ++i)
	max_buffer_len_ = std::max(max_buffer_len_, sizes->at(i));
      max_buffer_len = std::max(max_buffer_len_, max_buffer_len);
#endif

      //raw output
      if(raw_output) {
        write_sample_to_file(*raw_files[simulation_id], sample_time, *sim_sample);
#ifdef LOG
	cerr << "raw output written for sim. #" << simulation_id << " @ " << sample_time << endl;
#endif
      }

      if(!push_counter) {
	while(true) {
	  frontier = true;
	  //if instant-sample is complete, send it out
	  for (unsigned int i=0; i< n_simulations; ++i)
	    frontier &= !(sq[i]->empty());
	  if(frontier) {
#ifdef DATASIZE
	    datasize += sizeof(window_task_t) + (size_t)n_simulations * (size_t)n_monitors * sizeof(multiplicityType);
	    gettimeofday(&time_misuration, NULL);
	    if(itime < 0) {
	      itime = (double)time_misuration.tv_sec + (double)time_misuration.tv_usec / 1000000.0;
	    }
	    else {
	      double rtime = (double)time_misuration.tv_sec + (double)time_misuration.tv_usec / 1000000.0 - itime;
	      thr = (double)datasize/rtime;
	    }
#endif
	    if(stats_enabled) {
	      //send out window-task
	      sample_t **instant_samples = (sample_t **)MALLOC(n_simulations * sizeof(sample_t *)); //to be deallocated
	      double *sample_time_;
	      tq->pop((void **)&sample_time_);
	      for(unsigned int j=0; j<n_simulations; j++) {
		sq[j]->pop((void **)&instant_samples[j]);
#ifdef DATASIZE
		sizes->at(j)--;
#endif
	      }
	      void *p = MALLOC(sizeof(window_task_t)); //to be deallocated
	      ff_send_out(new (p) window_task_t(*sample_time_, instant_samples));
	      delete sample_time_;
	    }
	    else {
	      //destroy samples
	      sample_t *s;
	      for(unsigned int j=0; j<n_simulations; j++) {
		sq[j]->pop((void **)&s);
		delete s;
	      }
	    }
	  }
	  else
	    //incomplete frontier
	    break;
	}//end while-loop over frontiers
      }//end if(frontier)
    }//end for-loop over tasks

    delete v;
    return GO_ON;
  }

#ifdef DATASIZE
  void print_datasize_stats() {
    cout << "max. buffer size: " << max_buffer_len << endl;
    cout << "datasize: " << (double)datasize/1048576 << " MBytes (" << datasize << " Bytes)"
	 << ", throughput: " << thr/1024 << " KBytes/s"
	 << endl;
    cout << "latency: " << itime - ctime << " s" << endl;
  }
#endif

  void prepare(svector<msg_t*>*& v, size_t len, const int sender=-1) {
    svector<msg_t*> * v2 = new svector<msg_t*>(len);
    assert(v2);
    for(size_t i=0;i<len;++i) {
      msg_t *m = new msg_t;
      assert(m);
      v2->push_back(m);
    }
    v = v2;
  }

  virtual void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
    svector<msg_t *> &sv = *v[0]; //the vector of messages
    task = new vector<out_task_t *>; //allocate the output (vector of out_task_t)
    vector<out_task_t *> *task_ = (vector<out_task_t *> *)task;
    unsigned int k = sv.size() / 2;
    for(unsigned int i=0; i<k; ++i) {
      void *out_task = sv[2*i]->getData(); //out_task_t
      void *data = sv[2*i + 1]->getData(); //multiplicityType *
      task_->push_back(static_cast<out_task_t *>(out_task));
      task_->at(i)->monitor = new sample_t(n_monitors);
      multiplicityType * mm = (multiplicityType *)data;
      task_->at(i)->monitor->assign(&mm[0], &mm[0] + n_monitors);
      task_->at(i)->msg = sv[2*i]; //to be deleted
      delete sv[2*i + 1];
    }
    delete &sv;
  }

private:
  unsigned int n_simulations;
  unsigned nHosts;
  unsigned int n_monitors;
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
#ifdef DATASIZE
  size_t max_buffer_len;
  size_t datasize;
  double thr;
  struct timeval time_misuration;
  double ctime, itime;
  vector<unsigned long> *sizes;
#endif
  unsigned int push_counter, bulk_size;

protected:
  const std::string name;
  const std::string address;
  transport_t * transp;
};
#endif
