// D: ./scwc -i examples/ecoli/low.cwc -s 1 -t 10 --n-worker-sites 1 --role 1 -n 10
//./scwc -i examples/ecoli/low.cwc -s 1 -t 10 --n-worker-sites 0 --role 0 -n 10
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
using namespace std;

#ifdef USE_FF_DISTR
#include <ff_distributed.hpp>
#endif

#include <ff/pipeline.hpp>
#ifdef USE_STAT_ACCEL
#include <ff_stat_accel.hpp>
#endif
using namespace ff;

#include <Driver.h>
#include <Monitor.h>
#include <ff_nodes.hpp>


#include <utils.h>



int main(int argc, char * argv[]) {
#ifdef TIME
  struct timeval time_misuration;
  /*
  gettimeofday(&time_misuration, NULL);
  double rtime = (double)time_misuration.tv_sec * 1e+03 + (double)time_misuration.tv_usec * 1e-03;
  */
  double rtime = get_xtime_from(0, time_misuration);
#endif

  cout << "*** This is CWC Simulator" << endl;

  cwc_parameters_t param;
  parse_parameters(param, argc, argv);
  int v = validate_parameters(param);
  if(v == -1) {
    //help
    cout << *(param.desc) << endl;
    exit(0);
  }
  else if(v == 0) {
    //error
    cerr << "detected errors while parsing parameters" << endl;
    exit(1);
  }

  bool verbose = param.verbose;

  //input file
  ifstream *infile = new ifstream(param.infname.c_str());
  if(!infile->good()) {
    delete infile;
    cerr << "Could not open file: " << param.infname << endl;
    exit(1);
  }

  //parse cwc model
  scwc::Driver driver;
  bool result = driver.parse_stream(*infile, param.infname.c_str());
  delete infile;
  if(!result) {
    //syntax error
    cerr << param.infname << ": syntax error." << endl;
    exit(1);
  }
  if(param.syntax) {
    cout << param.infname << ": syntax ok." << endl;
    exit(0);
  }

  //get number of monitors
  unsigned int n_monitors = driver.model->monitors.size();
  
  //get lables
  string model_label = driver.model->title;
  vector<string> labels(n_monitors);
  for(unsigned int i=0; i<n_monitors; ++i)
    labels[i] = (driver.model->monitors)[i]->title;

  //adjust timing values
  int n_samples = (int)ceil(param.time_limit / param.sampling_period);
  param.time_limit = n_samples * param.sampling_period;

#ifdef USE_FF_ACCEL
  //farm. acc parameters
  int samples_per_slice = n_samples / param.n_slices;
#endif

#ifdef USE_STAT_ACCEL
  if(param.n_stat_workers < 1)
      param.n_stat_workers = n_monitors;
#endif

  //print verbose headers
  if(verbose) {
    cout << "input file: " << param.infname << endl
	 << "output prefix: " << param.outfile_prefix << endl;
    if(param.raw_output) cout << "raw output enabled" << endl;
    cout << "Model: " << driver.model->title << endl;
    if(param.fixed_seed)
      cout << "fixed seed: " << param.fixed_seed_value << endl;
    cout << "Time-limit: " << param.time_limit << endl;
    cout << "Number of samples: " << n_samples << endl;
#ifdef HYBRID
    cout << "Hybrid (stochastic/deterministic) semantics" << endl;
    if(param.rate_cutoff != RATE_CUTOFF_INFINITE)
      cout << "Minimum rate cutoff: " << param.rate_cutoff << endl;
    else
      cout << "Minimum rate cutoff: " << "infinite" << endl;
    if(param.population_cutoff != POPULATION_CUTOFF_INFINITE)
      cout << "Minimum population cutoff: " << param.population_cutoff << endl;
    else
      cout << "Minimum population cutoff: " << "infinite" << endl;
#else
    cout << "Pure stochastic semantics" << endl;
#endif
    //stats
    cout << "---\nStatistical parameters:" << endl;
    if(param.stats_enabled) {
      cout << "n. runs: " << param.n_simulations << endl
	   << "n. quantiles: " << param.n_quantiles << endl
	   << "n. clusters (k-means): " << param.n_clusters << endl
	   << "clustering threshold (QT): " << param.qt_threshold << endl
	   << "Grubb's test level: " << param.grubb << "%" << endl
	   << "interpolation-window size: " << param.window_size << endl
	   << "prediction factor: " << param.prediction_factor << endl
	   << "peak-detection parameters: " << param.peaks_parameter1
	   << ", " << param.peaks_parameter2 << endl;
    }
    else
      cout << "stats disabled" << endl;
#ifdef USE_FF_ACCEL
    //sim. farm-acc
    cout << "---\nFF sim. farm-acc. parameters:" << endl
	 << "n. sim. workers: " << param.n_workers << endl
	 << "n. scheduling slices: " << param.n_slices << endl
	 << "inflight: " << param.p_inflight << "%" << endl;
#endif
#ifdef USE_STAT_ACCEL
    //stat. farm-acc.
    cout << "---\nFF stat. farm-acc. parameters:" << endl
	 << "n. stat. workers: " << param.n_stat_workers << endl;
#endif
  }
  



#ifdef USE_FF_DISTR
  /*
    ------------------------------------------------------------------------------
    DISTRIBUTED
    ------------------------------------------------------------------------------
  */
  if(verbose) cout << "---\nDistributed mode" << endl;

  // creates the network using 0mq as transport layer
  zmqTransport transport(param.role? -1 : param.nhosts);
  if (transport.initTransport()<0) abort();

  //double running_time = -1;
  if (param.role) {
    //MASTER
    if(verbose)
      cout << "MASTER (n.hosts: " << param.nhosts << ")" << endl
	   << "Channels:" << endl
	   << "SCATTER: " << param.scatter_address << endl
	   << "FROMANY: " << param.fromany_address << endl
	   << "---\n"; 

    unsigned int n_simulations_min = param.nhosts;
#ifdef USE_FF_ACCEL
    n_simulations_min *= param.n_workers;
#endif

    if(param.n_simulations < n_simulations_min) {
      param.n_simulations = n_simulations_min;
      cerr << "number of simulations adjusted to " << param.n_simulations << endl;
    }

    // This should be generalized
    if(param.n_simulations%(n_simulations_min)!=0) {
      param.n_simulations = ((param.n_simulations/(n_simulations_min))+1)*(n_simulations_min);
      cerr << "number of simulations adjusted to " << param.n_simulations << endl;
    }

    //sim. farm
    SeedsProducer E(param, "SeedsEmitter", &transport);
    TrajectoriesConsumer C(param, n_monitors, n_samples, "TrajectoriesConsumer", &transport);
    ff_pipeline main_pipe;
    ff_pipeline sim_farm;
    sim_farm.add_stage(&E);
    sim_farm.add_stage(&C);
    main_pipe.add_stage(&sim_farm);

    //analysis pipe
    WindowsGenerator *W = NULL;
    StatEngine *S = NULL;
    ff_pipeline *analysis_pipe = NULL;
    if(param.stats_enabled) {
      W = new WindowsGenerator(param, n_monitors);
      S = new StatEngine(param, n_monitors, model_label, labels);
      analysis_pipe = new ff_pipeline(false, 1024, 1024, false);
      analysis_pipe->add_stage(W);
      analysis_pipe->add_stage(S);
      main_pipe.add_stage(analysis_pipe);
    }

    cout << "> Ready" << endl;

    if (main_pipe.run_and_wait_end()<0) {
      error("running pipeline\n");
      return -1;
    }
#ifdef TIME
    if(verbose) {
      cout << "* TIME stats ---\n"
	   << "- Sim. Farm\n"
	   << "running time: " << sim_farm.ffwTime()/1000 << " s" << endl;
      if(param.stats_enabled) {
	cout << "- Analysis Pipe\n"
	     << "running time: " << analysis_pipe->ffwTime()/1000 << " s" << endl;
	cout << "*\n";
      }
    }
#endif
#ifdef DATASIZE
  if(verbose) {
    cout << "- Alignment stage" << endl;
    C.print_datasize_stats();
    if(param.stats_enabled) {
      cout << "- Windows stage" << endl;
      W->print_datasize_stats();
    }
    cout << "*\n";
  }
#endif

    //clean-up
    if(S)
      delete S;
    if(W)
      delete W;
    if(analysis_pipe)
      delete analysis_pipe;

  } else {
    //SLAVE
    if(verbose)
      cout << "SLAVE (host # " << param.nhosts << ")" << endl
	   << "Channels:" << endl
	   << "SCATTER: " << param.scatter_address << endl
	   << "FROMANY: " << param.fromany_address << endl
	   << "---\n"; 

    ff_pipeline sim_pipe;
    TrajectoriesProducer T(param, n_monitors, "SimulationEngine", &transport);
    SeedsConsumer I(param, &T, "InputParser", &transport);
    SimEngine S(
		param,
#ifdef USE_FF_ACCEL
		samples_per_slice,
#endif
		n_samples, n_monitors
		);
    
    sim_pipe.add_stage(&I);
    sim_pipe.add_stage(&S);
    sim_pipe.add_stage(&T);

    if (!RECYCLE.init()) abort();

    cout << "> Ready" << endl;

    if (sim_pipe.run_and_wait_end()<0) {
      error("running pipeline\n");
      return -1;
    }

#ifdef TIME
    if(verbose) {
      cout << "* TIME stats ---\n"
	   << "- Sim. Pipe\n"
	   << "running time: " << sim_pipe.ffwTime()/1000 << " s" << endl;
    }
#endif
#ifdef DATASIZE
  if(verbose) {
    cout << "* DATASIZE stats ---\n"
	 << "- Sim. stage" << endl;
    S.print_datasize_stats();
    cout << "*\n";
  }
#endif
  }

  transport.closeTransport();





#else //defined(USE_FF_DISTR)
  /*
    ------------------------------------------------------------------------------
    SHARED MEMORY
    ------------------------------------------------------------------------------
  */
  if(verbose) cout << "---\nShared-memory mode" << endl;

  //sim. pipe
  TasksGenerator E(param, EMITTER_RANK);
  SimEngine Sm(
	       param,
#ifdef USE_FF_ACCEL
	       samples_per_slice,
#endif
	       n_samples, n_monitors, SIMENGINE_RANK
	       );
  TrajectoriesAlignment T(param, n_monitors, n_samples, ALIGNMENT_RANK);
  ff_pipeline main_pipe(false, 1024, 1024, false);
  ff_pipeline sim_pipe(false, 1024, 1024, false);
  sim_pipe.add_stage(&E);
  sim_pipe.add_stage(&Sm);
  sim_pipe.add_stage(&T);
  main_pipe.add_stage(&sim_pipe);

  //analysis pipe
  WindowsGenerator *W = NULL;
  StatEngine *St = NULL;
  ff_pipeline *analysis_pipe = NULL;
  if(param.stats_enabled) {
    W = new WindowsGenerator(param, n_monitors, WINDOWS_RANK);
    St = new StatEngine(param, n_monitors, model_label, labels, STATS_RANK);
    analysis_pipe = new ff_pipeline(false, 1024, 1024, false);
    analysis_pipe->add_stage(W);
    analysis_pipe->add_stage(St);
    main_pipe.add_stage(analysis_pipe);
  }
  

  cout << "> Ready" << endl;

  if (main_pipe.run_and_wait_end()<0) {
    error("running pipeline\n");
    return -1;
  }
#ifdef TIME
  if(verbose) {
    cout << "* TIME stats ---\n"
	 << "- Sim. Pipe (with sim. details)\n"
	 << "running time: " << sim_pipe.ffwTime()/1000 << " s" << endl;
    Sm.print_time_stats();
    T.print_time_stats();
#endif
#ifdef GRAIN
    cout << "* GRAIN stats" << endl;
    Sm.print_grain_stats();
#endif
#ifdef TIME
    if(param.stats_enabled) {
      cout << "- Analysis Pipe\n"
	   << "running time: " << analysis_pipe->ffwTime()/1000 << " s" << endl;
      W->print_time_stats();
      St->print_time_stats();
      
      cout << "*\n";
    }
  }
#endif
#ifdef DATASIZE
  if(verbose) {
    cout << "* DATASIZE stats ---\n"
	 << "- Sim. stage" << endl;
    Sm.print_datasize_stats();
    cout << "- Alignment stage" << endl;
    T.print_datasize_stats();
    if(param.stats_enabled) {
      cout << "- Windows stage" << endl;
      W->print_datasize_stats();
    }
    cout << "*\n";
  }
#endif

  //clean-up
  if(St)
    delete St;
  if(W)
    delete W;
  if(analysis_pipe)
    delete analysis_pipe;
#endif //defined(USE_FF_DISTR)

  std::cout << "done\n";
#ifdef TIME
  /*
  gettimeofday(&time_misuration, NULL);
  rtime = (double)time_misuration.tv_sec * 1e+03 + (double)time_misuration.tv_usec * 1e-03 - rtime;
  */
  rtime = get_xtime_from(rtime, time_misuration);
#ifdef USE_FF_DISTR
  if(param.role) {
#endif
    cout << rtime/1000 << " s" << endl;
    cout << rtime << endl;
#ifdef USE_FF_DISTR
  }
#endif
#endif //ifdef TIME
  return 0;
}
