#ifndef _CWC_PARAMETERS_HPP_
#define _CWC_PARAMETERS_HPP_

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#ifdef USE_FF_ACCEL
#include <ff/mapping_utils.hpp>
#endif

//default parameters
#ifdef HYBRID
#include "ode.h"
#define RATE_CUTOFF_DEFAULT RATE_CUTOFF_INFINITE //to be changed
#define POPULATION_CUTOFF_DEFAULT POPULATION_CUTOFF_INFINITE //to be changed
#endif
//statistics parameters
#define N_SIMULATIONS_MIN 1
#define WINDOW_SIZE_DEFAULT 11
#define GRUBB_DEFAULT 50
#define N_QUANTILES_DEFAULT 5
#define N_CLUSTERS_DEFAULT 2
#define DEGREE_DEFAULT 2
#if defined(USE_FF_ACCEL)
//sim. farm-acc. parameters
#define N_SLICES_DEFAULT 10
#define P_INFLIGHT_DEFAULT 50 //must be < 100
//#define P_WAITING_APPROX 13
//#define WAITING_THRESHOLD 10000 //microseconds
#endif
#ifdef USE_FF_DISTR
#define SCATTER_ADDRESS_DEFAULT "localhost:5000"
#define FROMANY_ADDRESS_DEFAULT "localhost:5001"
#endif

typedef struct cwc_parameters_t {
  po::variables_map vm;
  po::options_description *desc;
  //global
  string infname, outfile_prefix;
  double time_limit, sampling_period;
  unsigned int n_simulations;
  int fixed_seed_value;
  bool stats_enabled, fixed_seed, raw_output, syntax, verbose;
#ifdef HYBRID
  //hybrid
  double rate_cutoff;
  multiplicityType population_cutoff;
#endif
  //statistics
  unsigned int window_size, degree;
  double prediction_factor;
  double grubb;
  int n_quantiles, n_clusters;
  double qt_threshold;
  double peaks_parameter1, peaks_parameter2;
#ifdef USE_FF_ACCEL
  //farm acc. for simulations
  int n_workers, n_slices, p_inflight;
#endif
#ifdef USE_STAT_ACCEL
  int n_stat_workers;
#endif
#ifdef USE_FF_DISTR
  //distributed
  int role;
  unsigned int nhosts;
  string scatter_address, fromany_address;
#endif

  cwc_parameters_t() {
    raw_output = fixed_seed = raw_output = verbose = syntax = false;
    stats_enabled = true;
  }

  ~cwc_parameters_t() {
    if(desc)
      delete desc;
  }
} cwc_parameters_t;

//parse function
void parse_parameters(cwc_parameters_t &p, int argc, char *argv[]) {
  p.desc = new po::options_description("Allowed options");
  p.desc->add_options()
    ("help,h", "this help message")
    ("input-file,i", po::value<string>(), "input file")
    ("output-file,o", po::value<string>(), "output file")
    ("syntax", "verify the syntax of the input file and exit") //only check syntax
    ("time-limit,t", po::value<double>(&p.time_limit)->default_value(0), "time limit")
    ("sampling-period,s", po::value<double>(), "sampling period")
    ("simulations,n", po::value<unsigned int>(&p.n_simulations)->default_value(N_SIMULATIONS_MIN), "number of simulations (should be > n*n-worker-sites)")
#ifdef HYBRID
    //hybrid
    ("rate-cutoff", po::value<double>(&p.rate_cutoff)->default_value(RATE_CUTOFF_DEFAULT), "minimum rate cutoff")
    ("population-cutoff", po::value<multiplicityType>(&p.population_cutoff)->default_value(POPULATION_CUTOFF_DEFAULT), "minimum population cutoff")
#endif
    //statistics
    ("window-size", po::value<unsigned int>(&p.window_size)->default_value(WINDOW_SIZE_DEFAULT), "size of the sampling window")
    ("degree", po::value<unsigned int>(&p.degree)->default_value(DEGREE_DEFAULT), "interpolation degree")
    ("prediction-factor", po::value<double>(&p.prediction_factor)->default_value(5), "prediction factor")
    ("grubb", po::value<double>(&p.grubb)->default_value(GRUBB_DEFAULT), "%-level for Grubb's test")
    ("quantiles,q", po::value<int>(&p.n_quantiles)->default_value(N_QUANTILES_DEFAULT), "number of quantiles (> 2)")
    ("clusters,c", po::value<int>(&p.n_clusters)->default_value(N_CLUSTERS_DEFAULT), "number of (K-means) clusters")
    ("qt-threshold", po::value<double>(), "QT-clustering threshold")
    ("peaks-p1", po::value<double>(&p.peaks_parameter1)->default_value(0.01), "Peaks detection: parameter 1")
    ("peaks-p2", po::value<double>(&p.peaks_parameter2)->default_value(0.01), "Peaks detection: parameter 2")
    ("raw-output,r", "raw output")
    ("no-stats", "don't compute statistics")
#ifdef USE_FF_ACCEL
    ("workers,w", po::value<int>(&p.n_workers)->default_value(ff_numCores()), "number of workers")
    ("slices", po::value<int>(&p.n_slices)->default_value(N_SLICES_DEFAULT), "number of time-limit fractions for scheduling")
    ("inflight", po::value<int>(&p.p_inflight)->default_value(P_INFLIGHT_DEFAULT), "% of inflight tasks")
#endif
#ifdef USE_STAT_ACCEL
    ("stat-workers", po::value<int>(&p.n_stat_workers)->default_value(-1), "number of stat. workers")
#endif
#ifdef USE_FF_DISTR
    ("role", po::value<int>(&p.role)->default_value(0),"the excutable role in the distributed version")
    ("n-worker-sites", po::value<unsigned int>(&p.nhosts)->default_value(1), "n of distributed machines computing simulations")
    ("address1", po::value<string>(&p.scatter_address)->default_value(SCATTER_ADDRESS_DEFAULT), "address of the Scatter channel") //Scatter
    ("address2", po::value<string>(&p.fromany_address)->default_value(FROMANY_ADDRESS_DEFAULT), "address of the FromAny channel") //FromAny
#endif
    ("fixed-seed,f", po::value<int>(&p.fixed_seed_value), "fix seeds")
    ("verbose,v", "verbose mode")
    ;
  po::store(po::parse_command_line(argc, argv, *(p.desc)), p.vm);
  po::notify(p.vm);
}

//output: -1 for help, 0 for error, 1 for ok 
int validate_parameters(cwc_parameters_t &p) {
  po::variables_map &vm(p.vm);

  //help
  if(vm.count("help"))
    return -1;

  //syntax-only
  if((p.syntax = vm.count("syntax"))) return 0;

  //verbose
  p.verbose = vm.count("verbose");

  //input filename
  if(vm.count("input-file"))
    p.infname = vm["input-file"].as<string>();
  else {
    cerr << "No input file provided" << endl;
    return 0;
  }

  //output filename-prefix
  if(vm.count("output-file"))
    p.outfile_prefix = vm["output-file"].as<string>();
  else {
    //default output-filename prefix
    size_t begin_op = min(p.infname.find_last_of('/'), p.infname.find_last_of('\\'));
    if(begin_op == string::npos)
      begin_op = 0;
    else
      ++begin_op;
    size_t end_op = p.infname.find_last_of('.');
    if(end_op != string::npos)
      end_op -= begin_op;
    p.outfile_prefix = p.infname.substr(begin_op, end_op);
  }

  //raw output
  p.raw_output = vm.count("raw-output");

  //time limit
  if(p.time_limit < 0) {
    cerr << "Negative time limit" << endl;
    return 0;
  }

  //sampling period
  if(vm.count("sampling-period"))
    p.sampling_period = vm["sampling-period"].as<double>();
  else {
    cerr << "No sampling period provided" << endl;
    return 0;
  }
  if(p.sampling_period <= 0) {
    cerr << "Invalid sampling period: " << p.sampling_period << endl;
    return 0;
  }
  if(p.sampling_period > p.time_limit) {
    cerr << "Sampling period must be <= than time limit" <<  endl;
    return 0;
  }

  //statistical parameters
  p.stats_enabled = !vm.count("no-stats");
  if(p.stats_enabled) {
    //number of quantiles
    if (p.n_quantiles < 3) {
      p.n_quantiles = N_QUANTILES_DEFAULT;
      cerr << "number of quantiles ajusted to " << p.n_quantiles << endl;
    }
    //number of clusters
    if (p.n_clusters < 1) {
      p.n_clusters = N_CLUSTERS_DEFAULT;
      cerr << "number of clusters ajusted to " << p.n_clusters << endl;
    }
    //qt threshold
    p.qt_threshold = vm.count("qt-threshold") ? vm["qt-threshold"].as<double>() : p.n_simulations / 2;
    //n. simulations
    if(p.n_simulations < 1) {
      p.n_simulations = N_SIMULATIONS_MIN;
      cerr << "number of simulations adjusted to " << p.n_simulations << endl;
    }
  }

  //fixed seed?
  p.fixed_seed = vm.count("fixed-seed");

#ifdef USE_FF_ACCEL
  if(0.01 * p.p_inflight * p.n_simulations < p.n_workers)
    p.p_inflight = (int)(std::min)(99.0, 100.0 * ceil((float)p.n_workers / p.n_simulations));
#endif

  return 1;
}
#endif
