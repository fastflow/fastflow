
#include <iostream>
#include <cmath>
using namespace std;

#include <ff/ff.hpp>
using namespace ff;

#define NOFFLOADS 10 //scheduling slices
#define POINTSPEROFFLOAD 5 //samples per slice
#define WINSIZE 5
#define RANDOMSTEPS 1000





/*
  ------------------------------------------------------------------------------
  TasksGenerator:
  . produce seeds
  - prepare and send sim. tasks
  ------------------------------------------------------------------------------
*/
//simulation task
struct input_task_t {
  input_task_t(int i) : id(i), progress(0) {}
  int id;
  double progress;
};

class TasksGenerator: public ff_node {
public:
  TasksGenerator(int nsims) :
  nsims(nsims)
  {}

  void * svc(void*) {
    //prepare sim. tasks and send them out
    vector<input_task_t *> *tasks = new vector<input_task_t *>(nsims);
    for(int i=0; i<nsims; i++) {
      tasks->at(i) = new input_task_t(i);
    }
    ff_send_out(tasks);
    return (void *)FF_EOS;
  }

private:
  int nsims;
};





/*
  ------------------------------------------------------------------------------
  Acc1Engine:
  - generate and send out samples
  ------------------------------------------------------------------------------
*/
typedef vector<double> sample_t; //samples (one for each monitor)
struct out_task_t {
  out_task_t(input_task_t *st, double t, sample_t *m) :
    task_id(st->id),
    progress(t),
    monitor(m) {}
  int task_id;
  double progress;
  sample_t *monitor;
};

class Acc1_Collector: public ff_node {
public:
  void *svc(void *task) {
    return task;
  }
};

class Acc1_Worker: public ff_node {
public:
  void * svc(void * task) {
    input_task_t * t = (input_task_t *)task;
    for(int i=0; i<POINTSPEROFFLOAD; ++i) {
      //waste some time to produce x
      double x = 0;
      for(int j=0; j<RANDOMSTEPS; ++j)
	x += rand() % 10;
      x /= (10 * RANDOMSTEPS);
      t->progress += 0.1;
      ff_send_out(new out_task_t(t, t->progress, new vector<double>(1, x)));
    }
    return GO_ON;
  }
};



class Acc1Engine: public ff_node {
public:
  Acc1Engine(int nworkers) :nworkers(nworkers) {}

  void* svc(void* task) {
    vector<input_task_t *> &input_tasks = *((vector<input_task_t *> *) task);
    nsims = input_tasks.size();
    
    //set the accelerator
    ff_farm farm(true, nsims);  
    vector<ff_node *> workers;
    for(int i=0; i<nworkers; i++)
      workers.push_back(new Acc1_Worker());
    farm.add_workers(workers);
    Acc1_Collector c;
    farm.add_collector(&c);
    farm.run();

    int offloaded = 0;
    while(offloaded < NOFFLOADS) {
      //offload
      for(int i=0; i<nsims; ++i)
	farm.offload(input_tasks[i]);
      ++offloaded;
      //collect and send out
      int collected = 0;
      void *data = NULL;
      while(collected < nsims * POINTSPEROFFLOAD) {
	if(farm.load_result(&data)) {
	  ++collected;
	  ff_send_out(data);
	}
      }
    }
    farm.offload((void *)FF_EOS);
    farm.wait();

    for(int i=0; i<nsims; ++i)
      delete input_tasks[i];
    delete &input_tasks;
    return GO_ON;
  }

private:
  int nsims, nworkers;
};





/*
  ------------------------------------------------------------------------------
  TrajectoriesAlignment:
  - send out samples
  ------------------------------------------------------------------------------
*/
typedef struct window_task_t {
  window_task_t(double t, sample_t **s):
    sample_time(t), samples(s) {}
  double sample_time;
  sample_t **samples;
} window_task_t;

class TrajectoriesAlignment: public ff_node {
public:
  TrajectoriesAlignment(int nsims) :
  n_simulations(nsims),
  n_samples(NOFFLOADS * POINTSPEROFFLOAD)
  {
    //sampling queues
    for(unsigned int i=0; i<n_simulations; ++i) {
      sq.push_back(new uSWSR_Ptr_Buffer(n_samples / 10));
      sq[i]->init();
    }
    tq = new uSWSR_Ptr_Buffer(n_samples / 10);
    tq->init();
    newest_sample_time = -1;
    sizes = new vector<unsigned long>(n_simulations,0);
    bulk_size = n_simulations * n_samples / NOFFLOADS;
    push_counter = 0;
  }

  void svc_end() {
    for(unsigned int i=0; i<n_simulations; ++i)
      delete sq[i];
    delete tq;
    delete sizes;
  }

  void* svc(void *task) {
    out_task_t *sample_task = (out_task_t *) task;
    int simulation_id = sample_task->task_id;
    double sample_time = sample_task->progress;
    sample_t *sim_sample = sample_task->monitor;
    delete sample_task;
    push_counter = (push_counter + 1) % bulk_size;

    //buffer sample
    sq[simulation_id]->push(sim_sample);

    if(sample_time > newest_sample_time) {
      double *ptime = (double *)malloc(sizeof(double));
      *ptime = sample_time;
      tq->push(ptime);
      newest_sample_time = sample_time;
    }
    ++sizes->at(simulation_id);

    if(!push_counter) {
      long unsigned int n_pop = (long unsigned int)n_samples;
      for (unsigned int i=0; i< n_simulations; ++i)
	n_pop = (std::min)(n_pop, (sizes->at(i)));
      while(n_pop--) {
	//send out window-task
	sample_t **instant_samples = (sample_t **)malloc(n_simulations * sizeof(sample_t *)); //to be deallocated
	void *sample_time_=NULL;
	tq->pop(&sample_time_);
	for(unsigned int j=0; j<n_simulations; j++) {
	  sq[j]->pop((void **)&instant_samples[j]);
	  sizes->at(j)--;
	}
	ff_send_out(new window_task_t(*(double*)sample_time_, instant_samples));
	free(sample_time_);
      }
    }
    return GO_ON;
  }

private:
  unsigned int n_simulations;
  int n_samples;
  //sampling queues:
  // - one for each sim.
  // - queue-vector is indexed by sampling instants
  // - queue items: samples (one per monitor) at instant t
  vector<uSWSR_Ptr_Buffer *> sq;
  uSWSR_Ptr_Buffer *tq;
  double newest_sample_time;
  vector<unsigned long> *sizes;
  unsigned int push_counter, bulk_size;
};





/*
  ------------------------------------------------------------------------------
  WindowsGenerator:
  - generate stat. windows
  INPUT: sample_t ** (array of samples - one per simulation - at instant t)
  OUTPUT: Stat_Window * (stat. window cenetered at t)
  ------------------------------------------------------------------------------
*/
/*
  --------------------------------------------------
  statistical window (add-only circular buffer)
  operations (read/write) are performed at the front
  uses hazard-pointers
  --------------------------------------------------
*/
#include <valarray>
template <typename T>
class Stat_Window {
public:
  Stat_Window(unsigned int window_size, unsigned int n_simulations) :
    window_size(window_size), n_simulations(n_simulations) {
    begin = 0; //first free
    end = -1; //last occupied
    fill = 0; //n. occupied
    window = vector<valarray<T> *>(window_size, NULL);
    window_times = vector<double *>(window_size, NULL);
  }

  double get_last_time() {
    return *window_times[end];
  }

  double time_back(int n) {
    if(n <= end)
      return *window_times[end - n];
    else
      return *window_times[window_size - (n - end)];
  }

  valarray<T> &front_sample() {
    return *window[end];
  }

  void add_front(T datum, int sim_id) {
    (*window[end])[sim_id] = datum;
  }

  void free_oldest() {
    if(full()) {
      delete window[begin];
      delete window_times[begin];
    }
  }

  void free_all() {
    if(begin != -1)
      for(int i=begin; i!=end; i=(i+1)%window_size) {
	delete window[i];
	delete window_times[i];
      }
  }

  void slide(double sample_time) {
    end = (end + 1) % window_size;
    window[end] = new valarray<T>((T)0, n_simulations);
    window_times[end] = new double(sample_time);
    if(fill < window_size)
      ++fill;
    else
      begin = (begin + 1) % window_size;
  }

  unsigned int size() {
    return fill;
  }

  bool full() {
    return fill == window_size;
  }

  valarray<T> &operator[] (unsigned int i) {
    return *window[(begin + i) % window_size];
  }

  friend ostream& operator<<(ostream &os, Stat_Window &w) {
    for(unsigned int j=0; j<w.size(); ++j) {
      for(unsigned int i=0; i<w.n_simulations; ++i)
	os << "|\t" << w[j][i];
      os << endl;
    }
    return os;
  }

private:
  unsigned int window_size;
  int begin, end;
  unsigned int fill;
  vector<double *> window_times; //sampling instants
  vector<valarray<T> *> window; //by instants
  unsigned int n_simulations;
};

class WindowsGenerator: public ff_node {
public:
  WindowsGenerator(int nsims) :
  n_simulations(nsims),
  n_monitors(1)
  {
    //stat windows
    for(int i=0; i<n_monitors; ++i)
      stat_windows.push_back(new Stat_Window<double>(WINSIZE, n_simulations));
  }

  void svc_end() {
    for(int i=0; i<n_monitors; ++i)
      delete stat_windows[i];
  }

  void * svc(void * task) {
    window_task_t *t = (window_task_t *)task;
    sample_t **instant_samples = t->samples;
    double sample_time = t->sample_time;
    t->~window_task_t();
    delete(t);
    
    //slide
    for(int i=0; i<n_monitors; i++)
      stat_windows[i]->slide(sample_time);

    //fill front of monitor-window
    for(int j=0; j<n_simulations; j++) {
      for(int i=0; i<n_monitors; i++)
	stat_windows[i]->add_front(instant_samples[j]->at(i), j);
      delete instant_samples[j];
    }
    delete(instant_samples);

    //send copies of stat. windows
    vector<Stat_Window<double> *> *out = new vector<Stat_Window<double> *>;
    for(int i=0; i<n_monitors; ++i) {
      out->push_back(new Stat_Window<double>(*stat_windows[i]));
    }
    return out;
  }

private:
  int n_simulations;
  int n_monitors;
  //statistical windows (one for each monitor)
  vector<Stat_Window<double> *> stat_windows;
};





/*
  ------------------------------------------------------------------------------
  Acc2Engine:
  - compute/write stats
  INPUT: Stat_Window<multiplicityType> * (array of stat. windows - one per monitor)
  ------------------------------------------------------------------------------
*/
typedef struct stat_task_t {
  Stat_Window<double> *window;
  int opcode;
  stat_task_t(Stat_Window<double> *w, int op) :
    window(w), opcode(op) {}
} stat_task_t;

typedef struct res_task_t {
  double time;
  int opcode;
  double value;
  res_task_t(double t, int o, double v) :
    time(t), opcode(o), value(v) {}
} res_task_t;

class Acc2_Worker : public ff_node {
public:
  void *svc(void *task) {
    stat_task_t *t = (stat_task_t *)task;
    Stat_Window<double> *w = t->window;
    int opcode = t->opcode;
    delete t;
    double time = w->get_last_time();
    valarray<double> &data = w->front_sample();
    double res = 0, mean = 0;
    switch(opcode) {
    case 0:
      //mean
      res = data.sum() / data.size();
      break;
    case 1:
      //variance
      mean = data.sum() / data.size();
      for(unsigned int i=0; i<data.size(); ++i)
	res += pow(data[i] - mean, 2);
      res /= (data.size() - 1);
      break;
    case 2:
      //max
#ifdef max
#undef max //MD workaround to avoid clashing with max macro in minwindef.h
#endif
      res = data.max();
      break;
    }
    return new res_task_t(time, opcode, res);
  }
};

class Acc2_Collector : public ff_node {
  void *svc(void *fftask) {
    res_task_t *t = (res_task_t *)fftask;
    cout << "[" << t->time << "] ";
    switch(t->opcode) {
    case 0:
      cout << "MEAN";
      break;
    case 1:
      cout << "VARIANCE";
      break;
    case 2:
      cout << "MAX";
      break;
    }
    cout << ": " << t->value << endl;
    delete t;
    return GO_ON;
  }
};

class Acc2Engine: public ff_node {
public:
  Acc2Engine(int nworkers)
  :
  n_monitors(1),
  n_stat_workers(nworkers)
  {
    stat_windows = NULL;
    St = new ff_farm(true); //set stat. farm-acc.
    for(int i=0; i<n_stat_workers; ++i)
      stat_workers.push_back(new Acc2_Worker());
    St->add_workers(stat_workers);
    C = new Acc2_Collector();
    St->add_collector(C);
  }

  void svc_end() {
    for(int i=0; i<n_monitors; ++i) {
      stat_windows->at(i)->free_all();
      delete stat_windows->at(i);
    }
    delete stat_windows;
    St->wait();
    delete St;
    delete C;
  }

  void * svc(void * task) {
    //free oldest entry of the windows
    if(stat_windows) {
      for(int i=0; i<n_monitors; ++i) {
	stat_windows->at(i)->free_oldest();
	delete stat_windows->at(i);
      }
      delete stat_windows;
    }
    stat_windows = (vector<Stat_Window<double> *> *)task;

    printf("******************************* BEGIN ************************\n");
    //farm-accelerated statistics
    St->run_then_freeze();
    for(int i=0; i<n_monitors; ++i) {
      St->offload(new stat_task_t(stat_windows->at(i), 0 /*mean*/));
      St->offload(new stat_task_t(stat_windows->at(i), 1 /*variance*/));
      St->offload(new stat_task_t(stat_windows->at(i), 2 /*max*/));
    }
    //synchronize
    St->offload((void *)FF_EOS);
    St->wait_freezing();

    printf("******************************* FINE ************************\n");
    return GO_ON;
  }

private:
  int n_monitors;
  vector<Stat_Window<double> *> *stat_windows; //the most recent stat. windows
  int n_stat_workers;
  ff_farm *St;
  Acc2_Collector *C;
  vector<ff_node *> stat_workers;
};





int main(int argc, char * argv[]) {
    int nworkers1 = 3;
    int nworkers2 = 5;
    int nsims = 3;
    if (argc>1) {
	if(argc < 4) {
	    cerr << "usage: " << argv[0] << " nworkers1 nworkers2 nsims" << endl;
	    exit(1);
	}
	nworkers1 = atoi(argv[1]);
	nworkers2 = atoi(argv[2]);
	nsims = atoi(argv[3]);
    }
  //sim. pipe
  TasksGenerator E(nsims);
  Acc1Engine Sm(nworkers1);
  TrajectoriesAlignment T(nsims);
  ff_pipeline main_pipe(false, 1024, 1024, false);
  ff_pipeline sim_pipe(false, 1024, 1024, false);
  sim_pipe.add_stage(&E);
  sim_pipe.add_stage(&Sm);
  sim_pipe.add_stage(&T);
  main_pipe.add_stage(&sim_pipe);

  //analysis pipe
  WindowsGenerator W(nsims);
  Acc2Engine St(nworkers2);
  ff_pipeline analysis_pipe(false, 1024, 1024, false);
  analysis_pipe.add_stage(&W);
  analysis_pipe.add_stage(&St);
  main_pipe.add_stage(&analysis_pipe);

  //run
  cout << "> Ready" << endl;
  if (main_pipe.run_and_wait_end()<0) {
    error("running pipeline\n");
    return -1;
  }
  cout << "done\n";
  return 0;
}
