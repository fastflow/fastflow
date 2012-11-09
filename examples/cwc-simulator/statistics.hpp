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

#ifndef _CWC_STATISTICS_HPP_
#define _CWC_STATISTICS_HPP_
#define ENABLE_GNUPLOT true
#define WINDOW_SIZE 11

#include <vector>
#include <algorithm>
//#include "definitions.h"
#include "kmeans.h"
#include "filter.h" //init clustering
#include "qt.h"
#include "gnuplot.h"
#include "utils.h"
#include "output.h"
#include <boost/math/distributions/students_t.hpp>
using namespace boost::math;
using namespace std;
using namespace cwc_utils;

class Curves_comparation {
public:
  Curves_comparation(unsigned int point_i) : point_i(point_i) {}

  bool operator()(const valarray<double> &curveA, const valarray<double> &curveB) {
    return curveA[point_i] < curveB[point_i];
  }

private:
  unsigned int point_i;
};





/*
  --------------------------------------------------
  statistical window (add-only circular buffer)
  operations (read/write) are performed at the front
  --------------------------------------------------
*/
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
    //++last_index;
    end = (end + 1) % window_size;
    window[end] = new valarray<multiplicityType>((T)0, n_simulations);
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

  size_t get_sizeof() {
    return (size_t)window_size *
      //(sizeof(double) + sizeof(valarray<T> *)); //shallow counting
    (sizeof(double) + (size_t)n_simulations * sizeof(T)); //deep counting
  }

private:
  unsigned int window_size;
  int begin, end;
  unsigned int fill;
  vector<double *> window_times; //sampling instants
  vector<valarray<T> *> window; //by instants
  unsigned int n_simulations;
};





/*
  -------------------
  statistical engines
  -------------------
*/
template <typename T>
class Statistical_Engine {
public:
  Statistical_Engine(const bool gnuplot) :
  gnuplot(gnuplot)
#ifdef TIME
  , rtime(0)
#endif
  {}

  virtual ~Statistical_Engine() {
    if(outfile) {
      outfile->close();
      delete outfile;
    }
    outfile = NULL;
  }

  virtual void init(string prefix, string postfix, string desc, string title, int n_simulations, double time_limit, double time_delta) = 0;
  virtual void compute_write(Stat_Window<T> &stat_window) = 0;
  virtual void finalize(Stat_Window<T> &stat_window) = 0;

#ifdef TIME
  double get_running_time() {
    return rtime;
  }
#endif

protected:
  ofstream *outfile;
  const bool gnuplot;
#ifdef TIME
  double rtime;
  //struct rusage usage;
  struct timeval usage;
#endif

  ofstream *try_open(string fname) {
    ofstream *of = new ofstream(fname.c_str());
    if(!of->is_open()) {
      cerr << "Could not open file: " << fname << endl;
      exit(1);
    }
    return of;
  }
};





/*
  -------------
  MEAN-VARIANCE
  -------------
*/
template <typename T>
class MeanVariance_Engine : public Statistical_Engine<T> {
public:
  MeanVariance_Engine(const double grubb = 0, bool gnuplot = ENABLE_GNUPLOT) :
    Statistical_Engine<T>(gnuplot), grubb(grubb) {}

  virtual void init(string prefix, string postfix, string desc, string title, int n_simulations, double time_limit, double time_delta) {
    //open file
    string fprefix = prefix + "_reductions_" + postfix;
    string fname = fprefix + ".dat";
    this->outfile = this->try_open(fname);
    //write headers
    stringstream header;
    header << "# Reductions for: " << desc << endl;
    header << "#" << endl;
    header << "# Sample mean and variance";
    if(grubb > 0)
      header << " (under " << grubb << "%-level Grubb's test)";
    header << endl;
    header << "#" << endl;
    header << "# Subject:" << title << endl;
    header << "#" << endl;
    header << "# time\tmean\tvariance" << endl;
    string s_header = header.str();
    write_to_file(*this->outfile, s_header);
    //write gnuplot
    if(this->gnuplot)
      gnuplot_script_reductions(fprefix, fname, title, time_limit, grubb * 100);
  }

  virtual void compute_write(Stat_Window<T> &stat_window) {
#ifdef TIME
    double t = get_xtime_from(0, this->usage);
#endif
    valarray<T> &sample(stat_window.front_sample());
    double time = stat_window.get_last_time();
    //compute
    vector<double> mv(2);
    pair<double, double> b_mv;
    if(grubb > 0)
      b_mv = grubb_mean_variance(sample, grubb);
    else 
      b_mv = mean_variance(sample);
    mv[0] = b_mv.first;
    mv[1] = b_mv.second;
    //write
    write_stat<double>(this->outfile, mv, time);
#ifdef TIME
    this->rtime += get_xtime_from(t, this->usage);
#endif
  }

  virtual void finalize(Stat_Window<T> &stat_window) {}

private:
  const double grubb;
  double sample_mean_sub(valarray<T> &sample, int lw, int up) {
    double acc = 0;
    for(int i=lw; i<=up; i++)
      acc += sample[i];
    return acc / (up - lw + 1);
  }

  double sample_variance_sub(double sample_mean, valarray<T> &sample, int lw, int up) {
    double sv = 0;
    for(int i=lw; i<=up; i++) {
      double error = sample[i] - sample_mean;
      sv += (error * error);
    }
    return (1 / double(up - lw)) * sv; //size - 1 = up - lw + 1 - 1
  }

  double sample_standard_deviation(double sample_variance) {
    return sqrt(sample_variance);
  }

  pair<double, double> mean_variance(valarray<T> &sample) {
    unsigned int size = sample.size();
    double mean = sample_mean_sub(sample, 0, size-1);
    double variance = sample_variance_sub(mean, sample, 0, size-1);
    return pair<double, double>(mean, variance);
  }

  pair<double, double> grubb_mean_variance(valarray<T> &sample, double grubb) {
    unsigned int n = sample.size();
    //test on:
    //h0: no outliers
    //ha: at least one outiler
    int limits[2], to_change, into;
    limits[0] = 0; //lw
    limits[1] = n - 1; //up
    double g_min, g_max, g, t2_q, ssd;
    double mean = 0;
    double variance = 0;
    while(true) {
      mean = sample_mean_sub(sample, limits[0], limits[1]);
      if(n <= 6)
	break;
      ssd = sample_standard_deviation(sample_variance_sub(mean, sample, limits[0], limits[1]));
      g_min = (mean - sample[limits[0]]) / ssd;
      g_max = (sample[limits[1]] - mean) / ssd;
      if(g_min > g_max) {
	g = g_min;
	to_change = 0;
	into = limits[0] + 1;
      }
      else {
	g = g_max;
	to_change = 1;
	into = limits[1] - 1;
      }
      students_t t(n - 2);
      t2_q = pow(quantile(t, grubb / (2 * n)), 2);
      double critical = (double(n - 1) / sqrt(double(n))) * sqrt(t2_q / (n - 2 + t2_q));
      if(g <= critical)
	//accept h0
	break;
      //refuse h0
      limits[to_change] = into;
      --n;
    }
    //n <= 6 or accepted h0
    variance = sample_variance_sub(mean, sample, limits[0], limits[1]);
    return pair<double, double>(mean, variance);
  }
};





/*
  ---------
  QUANTILES
  ---------
*/
template <class T>
class Quantiles_Engine : public Statistical_Engine<T> {
public:
  Quantiles_Engine(const int nq, bool gnuplot = ENABLE_GNUPLOT) :
    Statistical_Engine<T>(gnuplot), n_quantiles(nq) {}

  virtual void init(string prefix, string postfix, string desc, string title, int n_simulations, double time_limit, double time_delta) {
    //compute quantiles indexes
    q_indexes = linspace_int(n_simulations, n_quantiles);
    //open file
    string fprefix = prefix + "_quantiles_" + postfix;
    string fname = fprefix + ".dat";
    this->outfile = this->try_open(fname);
    //write headers
    //vector<int> q_indexes = linspace_int(n_simulations, n_quantiles);
    stringstream header;
    header << "# Quantiles for: " << desc << endl;
    header << "#" << endl;
    header << "# Quantiles: ";
    for(unsigned int i=0; i<n_quantiles; i++)
      header << (float(q_indexes[i])/(n_simulations - 1)) * 100 << "%, ";
    header << "#" << endl;
    header << "# Subject:" << title << endl;
    header << "#" << endl;
    //# time q. 0% q. 25% q. 50% q. 75% q. 100%
    /*
    header << "# Columns:" << endl
	   << "# 1. time" << endl;
    for(unsigned int j=0; j<n_quantiles; j++)
      header << "# " << j+2 << ". quantile " << (float(q_indexes[j])/(n_simulations - 1)) * 100 << "%" << endl;
    */
    header << "# time";
    for(unsigned int j=0; j<n_quantiles; j++)
      header << "\tq. " << (float(q_indexes[j])/(n_simulations - 1)) * 100 << "%";
    string s_header = header.str();
    //s_header.erase(s_header.length() - 1);
    write_to_file(*this->outfile, s_header + "\n");
    //write gnuplot
    if(this->gnuplot)
      gnuplot_script_classifications(fprefix, fname, title, time_limit, n_simulations/*, n_quantiles*/, q_indexes);
  }

  virtual void compute_write(Stat_Window<T> &stat_window) {
#ifdef TIME
    double t = get_xtime_from(0, this->usage);
#endif
    valarray<T> &sample_(stat_window.front_sample());
    double time = stat_window.get_last_time();
    vector<T> sample(sample_.size());
    for(unsigned int i=0; i<sample.size(); ++i)
      sample[i] = sample_[i];
    sort(sample.begin(), sample.end());
    //compute
    vector<T> q(n_quantiles);
    quantiles(sample, q);
    //write
    write_stat<T>(this->outfile, q, time);
#ifdef TIME
    this->rtime += get_xtime_from(t, this->usage);
#endif
  }

  virtual void finalize(Stat_Window<T> &stat_window) {}
  
private:
  const unsigned int n_quantiles;
  vector<int> q_indexes;
  void quantiles(vector<T> &sample, vector<T> &q) {
    for(unsigned int i=0; i<n_quantiles; i++)
      q[i] = sample[q_indexes[i]];
  }
};





/*
  --------------------------------------
  filter window (for clustering methods)
  --------------------------------------
*/
typedef struct filter_window {
  unsigned int left, right;

  filter_window() {
    left = right = 0;
  }

  filter_window(unsigned int size) {
    left = 0;
    right = size - 1;
  }

  void slide() {
    ++left;
    --right;
  }

  bool centered() {
    return left == right;
  }
} filter_window_t;





/*
  ------------------
  K-MEANS CLUSTERING
  ------------------
*/
/*
ostream &operator<<(ostream &os, centroid_t c) {
  return os << c.position << "\t" << c.size;
}
*/

template <class T>
class Kmeans_Engine : public Statistical_Engine<T> {
public:
  Kmeans_Engine(const int nc, const unsigned int window_size, bool gnuplot = ENABLE_GNUPLOT, const double step = 0, const unsigned int degree = 0, const double prediction_factor = 0) :
    Statistical_Engine<T>(gnuplot), window_step(step), degree(degree), prediction_factor(prediction_factor), n_clusters(nc), first_pass(true)
  {
    fw = filter_window(window_size);
  }

  virtual void init(string prefix, string postfix, string desc, string title, int n_simulations, double time_limit, double time_delta) {
    //open file
    string fprefix = prefix + "_kmeans_" + postfix;
    string fname = fprefix + ".dat";
    this->outfile = this->try_open(fname);
    //write headers
    stringstream header;
    header << "# K-means clusters for: " << desc << endl;
    header << "#" << endl;
    header << "# K-means clustering (centroid, predictions and sizes)" << endl;
    header << "#" << endl;
    header << "# Subject:" << title << endl;
    header << "#" << endl;
    /*
    header << "# Columns:" << endl
	   << "# 1. time" << endl;
    for(unsigned int j=0; j<n_clusters; j++)
      header << "# " << 2+j*3 << ". cluster " << j+1 << " (centroid)" << endl
	     << "# " << 2+j*3+1 << ". cluster " << j+1 << " (prediction)" << endl
	     << "# " << 2+j*3+2 << ". cluster " << j+1 << " (size)" << endl;
    */
    //# time c1 p1 s1
    header << "# time";
    for(unsigned int j=0; j<n_clusters; j++)
      header << "\tc" << 1+j << "\tp" << 1+j << "\ts" << 1+j;
    string s_header = header.str();
    //s_header.erase(s_header.length() - 1);
    write_to_file(*this->outfile, s_header + "\n");
    //write gnuplot
    if(this->gnuplot)
      gnuplot_script_kmeans(fprefix, fname, title, time_limit, n_clusters, time_delta, n_simulations); //with trends
  }

  void gnuplot_all(string prefix, string postfix, double time_limit, unsigned int monitor_id, unsigned int n_simulations) {
    if(this->gnuplot) {
      string kmeans_fname = prefix + "_kmeans_" + postfix + ".dat";
      string raw_prefix = prefix + "_s";
      gnuplot_script_all(prefix + "_all_" + postfix, kmeans_fname, raw_prefix, time_limit, monitor_id, n_simulations, n_clusters);
    }
  }

  virtual void compute_write(Stat_Window<T> &stat_window) {
#ifdef TIME
    double t = get_xtime_from(0, this->usage);
#endif
    bool full = stat_window.full();
    unsigned int window_size = stat_window.size();

    if(full) {
      //build curves
      unsigned int n_simulations = stat_window[0].size();
      vector<valarray<double> > curves(n_simulations, valarray<double>(window_size));
      for(unsigned int i=0; i<n_simulations; ++i)
	for(unsigned int j=0; j<window_size; ++j)
	  curves[i][j] = (double)stat_window[j][i];

      if(first_pass) {
	init_clustering(curves, stat_window); //init clustering and slide the window to center
	first_pass = false;
      }

      //clustering
      kmeans(
	     curves,
	     centroids,
	     points,
	     window_step,
	     window_size,
	     degree,
	     prediction_factor,
	     fw.left,
	     fw.right
	     );

      //write
      write_stat_notzero(this->outfile, centroids, stat_window.time_back(fw.right));
    }
#ifdef TIME
    this->rtime += get_xtime_from(t, this->usage);
#endif
  }

  virtual void finalize(Stat_Window<T> &stat_window) {
#ifdef TIME
    double t = get_xtime_from(0, this->usage);
#endif
    //build curves
    unsigned int window_size = stat_window.size();
    unsigned int n_simulations = stat_window[0].size();
    vector<valarray<double> > curves(n_simulations, valarray<double>(window_size));
    for(unsigned int i=0; i<n_simulations; ++i)
      for(unsigned int j=0; j<window_size; ++j)
	curves[i][j] = (double)stat_window[j][i];

    while(fw.right != 0) {
      fw.slide();

      //*this->outfile << "#filter-window: (" << fw.left << ", " << "1" << ", " << fw.right << ")" << endl;
      //clustering
      kmeans(
	     curves,
	     centroids,
	     points,
	     window_step,
	     window_size,
	     degree,
	     prediction_factor,
	     fw.left,
	     fw.right
	     );

      //write
      write_stat_notzero(this->outfile, centroids, stat_window.time_back(fw.right));
    }

#ifdef TIME
    this->rtime += get_xtime_from(t, this->usage);
#endif
  }
  
private:
  const double window_step;
  const unsigned int degree;
  const double prediction_factor;
  const unsigned int n_clusters;
  filter_window_t fw;
  bool first_pass;
  vector<centroid_t> centroids;
  vector<point_t> points;

  void init_clustering(vector<valarray<double> > &curves, Stat_Window<T> &stat_window) {
    unsigned int n_simulations = curves.size();
    vector<valarray<double> > s_curves(curves);
    Curves_comparation c(0); //compare by the first point
    sort(s_curves.begin(), s_curves.end(), c);
    //init centroids
    Savgol_Prediction savgol(0, fw.right, degree, prediction_factor * window_step, window_step);
    centroids = vector<centroid_t>(n_clusters);
    vector<int> init_ci = linspace_int(n_simulations, n_clusters);
    for(unsigned int i=0; i<n_clusters; ++i) {
      centroids[i].position = s_curves[init_ci[i]][0];
      centroids[i].prediction = savgol.prediction(s_curves[init_ci[i]]);
#ifdef LOG
      //cerr << "centroid " << i << ": " << centroids[i].position << "(p: " << centroids[i].prediction << ")" << endl;
#endif
    }

    //init memberships
    points = vector<point_t>(n_simulations);
    for(unsigned int i=0; i<n_simulations; ++i) {
      points[i].position_filtered = savgol.filt(curves[i]);
      points[i].position = curves[i][0];
      points[i].prediction = savgol.prediction(curves[i]);
      unsigned int ci = find_nearest_cluster(points[i], centroids, n_clusters);
      points[i].membership = ci;
      centroids[ci].size += 1;
#ifdef LOG
      //cerr << "point " << i << ": {membership = " << points[i].membership << ", position = " << points[i].position << ", prediction = " << points[i].prediction << "}" << endl;
#endif
    }

    /*
    //first-time set centroids and points into stat-window
    stat_window.set_centroids(centroids);
    stat_window.set_points(points);
    */

    //compute stats until filter-window is balanced
    //unsigned int w = 0;
    unsigned int window_size = stat_window.size();
    while(!fw.centered()) {
      //*this->outfile << "#filter-window: (" << fw.left << ", " << "1" << ", " << fw.right << ")" << endl;
      //clustering
      kmeans(
	     curves,
	     centroids,
	     points,
	     window_step,
	     window_size,
	     degree,
	     prediction_factor,
	     fw.left,
	     fw.right
	     );

      //write
      write_stat_notzero(this->outfile, centroids, (std::max)(0.0, stat_window.time_back(fw.right)));

      fw.slide();
    }
  }
};





/*
  -------------
  QT CLUSTERING
  -------------
*/
template <class T>
class QT_Engine : public Statistical_Engine<T> {
public:
  QT_Engine(const double t, const unsigned int window_size, bool gnuplot = ENABLE_GNUPLOT, const double step = 0, const unsigned int degree = 0, const double prediction_factor = 0) :
    Statistical_Engine<T>(gnuplot), threshold(t), window_step(step), degree(degree), prediction_factor(prediction_factor), first_pass(true), max_n_clusters(0)
  {
    fw = filter_window(window_size);
  }

  virtual void init(string prefix, string postfix, string desc, string title, int n_simulations, double time_limit, double time_delta) {
    //open file
    string fprefix = prefix + "_qt_" + postfix;
    dat_fname = fprefix + ".dat";
    this->outfile = this->try_open(dat_fname);
    //write headers
    stringstream header;
    header << "# QT clusters for: " << desc << endl;
    header << "#" << endl;
    header << "# QT clustering (centroids, sizes)" << endl;
    header << "#" << endl;
    header << "# Subject:" << title << endl;
    header << "#" << endl;
    /*
    header << "# 1. time" << endl
	   << "# 2. cluster 1 (center)" << endl
	   << "# 3. cluster 1 (size)" << endl
	   << "# ..." << endl
	   << "# 2*i. cluster i (center)" << endl
	   << "# 2*i + 1. cluster i (size)" << endl;
    */
    header << "# time clusters" << endl;
    write_to_file(*this->outfile, header.str());
    //write gnuplot (only headers)
    gnuplot_script_qt_init(&gnuplot_file, fprefix, dat_fname, title, time_limit, time_delta, n_simulations);
  }

  virtual void compute_write(Stat_Window<T> &stat_window) {
#ifdef TIME
    double t = get_xtime_from(0, this->usage);
#endif
    bool full = stat_window.full();
    unsigned int window_size = stat_window.size();

    if(full) {
      //build curves (rotate the window: from by-instants to by-simulations)
      unsigned int n_simulations = stat_window[0].size();
      vector<valarray<double> > curves(n_simulations, valarray<double>(window_size));
      for(unsigned int i=0; i<n_simulations; ++i)
	for(unsigned int j=0; j<window_size; ++j)
	  curves[i][j] = (double)stat_window[j][i];

      if(first_pass) {
	while(!fw.centered()) {
	  //clustering
	  qt_result_t c = qt_clustering(
					curves, threshold, window_step, stat_window.size(),
					degree,
					prediction_factor,
					fw.left,
					fw.right
					);
	  qt_center_t &res = c.first;
	  vector <list <int> > &res2 = c.second;
	  max_n_clusters = (std::max)(max_n_clusters, (int)c.first.size());

	  //write
	  write_stat_qt(this->outfile, res, res2, (std::max)(0.0, stat_window.time_back(fw.right)));
	  fw.slide();
	}
	first_pass = false;
      }

      qt_result_t c = qt_clustering(
				    curves, threshold,window_step, stat_window.size(),
				    degree,
				    prediction_factor,
				    fw.left,
				    fw.right
				    );
      qt_center_t res = c.first;
      vector <list <int> > res2 = c.second;
      max_n_clusters = (std::max)(max_n_clusters, (int)c.first.size());

      //write
      write_stat_qt(this->outfile, res, res2, stat_window.time_back(fw.right));
    }
#ifdef TIME
    this->rtime += get_xtime_from(t, this->usage);
#endif
  }

  virtual void finalize(Stat_Window<T> &stat_window) {
#ifdef TIME
    double t = get_xtime_from(0, this->usage);
#endif
    unsigned int window_size = stat_window.size();
    unsigned int n_simulations = stat_window[0].size();
    vector<valarray<double> > curves(n_simulations, valarray<double>(window_size));
    for(unsigned int i=0; i<n_simulations; ++i)
      for(unsigned int j=0; j<window_size; ++j)
	curves[i][j] = (double)stat_window[j][i];
      
    while(fw.right != 0) {
      fw.slide();

      //clustering
      qt_result_t c = qt_clustering(
				    curves,
			            threshold,
				    window_step,
				    stat_window.size(),
				    degree,
				    prediction_factor,
				    fw.left,
				    fw.right
				    );
      qt_center_t res = c.first;
      vector <list <int> > res2 = c.second;
      max_n_clusters = (std::max)(max_n_clusters, (int)c.first.size());
      
      //write
      write_stat_qt(this->outfile, res, res2, stat_window.time_back(fw.right));
    }

    gnuplot_script_qt_finalize(gnuplot_file, dat_fname, max_n_clusters);
#ifdef TIME
    this->rtime += get_xtime_from(t, this->usage);
#endif
  }

private:
  const double threshold;
  const double window_step;
  const unsigned int degree;
  const double prediction_factor;
  filter_window_t fw;
  bool first_pass;
  int max_n_clusters;
  ofstream *gnuplot_file;
  string dat_fname;

  qt_result_t qt_clustering(vector<valarray<double> > &curves, double threshold, double window_step, unsigned int window_size, unsigned int degree, double prediction_factor, unsigned int fwl, unsigned int fwr) {
    return qt(curves, threshold, window_step, window_size, degree, prediction_factor,fwl,fwr);
  }
};





/*
  ---------------
  Peaks detection
  ---------------
*/
template <class T>
class Peaks_Engine : public Statistical_Engine<T> {
public:
  Peaks_Engine(const unsigned int window_size, const double c1, const double c2, const unsigned int n_simulations, bool gnuplot, const double step, const unsigned int degree, const double prediction_factor) :
    Statistical_Engine<T>(gnuplot), window_step(step), degree(degree), prediction_factor(prediction_factor), c1(c1), c2(c2)
  {
    fw = filter_window(window_size);
    res = vector<double>(1);
    freqs = vector<double>(n_simulations, 0);
    last_peak_times = vector<double>(n_simulations, 0);
  }

  virtual void init(string prefix, string postfix, string desc, string title, int n_simulations, double time_limit, double time_delta) {
    //open file
    string fprefix = prefix + "_peaks_" + postfix;
    string fname = fprefix + ".dat";
    this->outfile = this->try_open(fname);
    //write headers
    stringstream header;
    header << "# Peaks for: " << desc << endl;
    header << "#" << endl;
    header << "# Subject:" << title << endl;
    header << "#" << endl;
    header << "# time\tfrequency";
    string s_header = header.str();
    s_header.erase(s_header.length() - 1);
    write_to_file(*this->outfile, s_header + "\n");
    //write gnuplot
    if(this->gnuplot)
      gnuplot_script_peaks(fprefix, fname, title, time_limit);
  }

  virtual void compute_write(Stat_Window<T> &stat_window) {
#ifdef TIME
    double t = get_xtime_from(0, this->usage);
#endif
    bool full = stat_window.full();
    unsigned int window_size = stat_window.size();

    if(full) {
      //build curves
      unsigned int n_simulations = stat_window[0].size();
      vector<valarray<double> > curves(n_simulations, valarray<double>(window_size));
      for(unsigned int i=0; i<n_simulations; ++i)
	for(unsigned int j=0; j<window_size; ++j)
	  curves[i][j] = (double)stat_window[j][i];

      while(true) {
	double t = (std::max)(0.0, stat_window.time_back(fw.right));
	res[0] = peaks(
		       curves, window_step, stat_window.size(),
		       degree,
		       prediction_factor,
		       fw.left,
		       fw.right,
		       t
		       );

	//write
	write_stat<double>(this->outfile, res, t);

	if(!fw.centered())
	  fw.slide();
	else
	  break;
      }
    }
#ifdef TIME
    this->rtime += get_xtime_from(t, this->usage);
#endif
  }

  virtual void finalize(Stat_Window<T> &stat_window) {
#ifdef TIME
    double t = get_xtime_from(0, this->usage);
#endif
    //build curves
    unsigned int window_size = stat_window.size();
    unsigned int n_simulations = stat_window[0].size();
    vector<valarray<double> > curves(n_simulations, valarray<double>(window_size));
    for(unsigned int i=0; i<n_simulations; ++i)
      for(unsigned int j=0; j<window_size; ++j)
	curves[i][j] = (double)stat_window[j][i];

    while(fw.right != 0) {
      fw.slide();

      double t = stat_window.time_back(fw.right);
      res[0] = peaks(
		     curves, window_step, stat_window.size(),
		     degree,
		     prediction_factor,
		     fw.left,
		     fw.right,
		     t
		     );

      //write
      write_stat<double>(this->outfile, res, t);
    }
#ifdef TIME
    this->rtime += get_xtime_from(t, this->usage);
#endif
  }
  
private:
  const double window_step;
  const unsigned int degree;
  const double prediction_factor;
  const double c1, c2;
  filter_window_t fw;
  vector<double> res, freqs, last_peak_times;

  double peaks(
	    vector<valarray<double> > &curves,
	    double time_win,
	    unsigned int window_size,
	    unsigned int degree,
	    double prediction_factor,
	    unsigned int fw_left,
	    unsigned int fw_right,
	    double current_time
	    )
  {
    Savgol_Prediction savgol(fw_left, fw_right, degree, prediction_factor * time_win, time_win);
    unsigned int n_simulations = curves.size();
    double sum = 0;
    for(unsigned int i=0; i<n_simulations; ++i) {
      if(savgol.peak(curves[i], c1, c2)) {
	freqs[i] = (double)1 / (current_time - last_peak_times[i]);
	last_peak_times[i] = current_time;
      }
      sum += freqs[i];
    }
    return (sum / n_simulations);
  }
};
#endif
