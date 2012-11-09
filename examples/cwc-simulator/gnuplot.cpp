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

#include "gnuplot.h"
#include "utils.h" //exit (should be <cstdlib>?)
#include "output.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
//using namespace cwc_utils;

#define GNUPLOT_SCRIPT_EXTENSION "gp"
//#define GNUPLOT_TERMINAL "postscript enhanced eps 20"
//#define GNUPLOT_OUTPUT_EXTENSION "eps"
#define GNUPLOT_TERMINAL "jpeg"
#define GNUPLOT_OUTPUT_EXTENSION "jpg"
#define N_PLOTTED_QUANTILES 3
#define POINT_SIZE 3

void gnuplot_init(stringstream &gnuplot_base, double time_limit) {
  gnuplot_base << "set grid" << endl;
  gnuplot_base << "set xlabel \"time\"" << endl;
  gnuplot_base << "set terminal " << GNUPLOT_TERMINAL << endl;
  gnuplot_base << "set xrange [0:" << time_limit << "]" << endl;
}





void gnuplot_script_reductions(
			       string prefix,
			       string fsource,
			       string title,
			       double time_limit,
			       double grubb
			       )
{
  //unsigned int n_monitors = monitors.size();
  stringstream gnuplot_base;
  gnuplot_init(gnuplot_base, time_limit);
  //for(unsigned int i=0; i<n_monitors; i++) {
  stringstream r_gnuplot;
  r_gnuplot << gnuplot_base.str();
  r_gnuplot << "set ylabel \"mean and standard deviation (" << title << ")\"" << endl;
  //file
  string out_fname = prefix + "." + GNUPLOT_SCRIPT_EXTENSION;
  ofstream f(out_fname.c_str());
  if(!f.is_open()) {
    cerr << "Could not open file: " << out_fname << endl;
    exit(1);
  }
  //script
  stringstream s;
  s << r_gnuplot.str();
  s << "set output \"" << prefix << "." << GNUPLOT_OUTPUT_EXTENSION << "\"" << endl;
  //standard deviation
  s << "plot \"" << fsource << "\" using 1:2:(sqrt($3))";
  s << " title \"standard deviation\""<< " with yerrorbars lt 2,\\" << endl;
  //mean
  s << "\"" << fsource << "\" using 1:2";
  s << " title \"mean";
  if(grubb > 0)
    s << " (under " << grubb << "%-grubb Grubb's test)";
  s << "\" with lines lw 2 lt 1" << endl;
  write_to_file(f, s.str());
  f.close();
}





void gnuplot_script_classifications(
				    string prefix,
				    string fsource,
				    string title,
				    double time_limit,
				    int n_simulations,
				    vector<int> &q_indexes
				    )
{
  unsigned int n_quantiles = q_indexes.size();
  //vector<int> q_indexes = linspace_int(n_simulations, n_quantiles);
  stringstream gnuplot_base;
  gnuplot_init(gnuplot_base, time_limit);
  stringstream c_gnuplot;
  c_gnuplot << gnuplot_base.str();
  c_gnuplot << "set ylabel \"quantiles (" << title << ")\"" << endl;
  //file
  string out_fname = prefix + "." + GNUPLOT_SCRIPT_EXTENSION;
  ofstream f(out_fname.c_str());
  if(!f.is_open()) {
    cerr << "Could not open file: " << out_fname << endl;
    exit(1);
  }
  //script
  stringstream s;
  s << c_gnuplot.str();
  s << "set output \"" << prefix << "." << GNUPLOT_OUTPUT_EXTENSION << "\"" << endl;
  //first non-zero quantile in selected range
  float q_level = (float(q_indexes[1])/(n_simulations - 1)) * 100;
  s << "plot \"" << fsource << "\" using 1:3 title \"" << q_level << "% quantile\" with lines";
  //all remaining quantiles but 100%
  for (unsigned int j=2; j<n_quantiles-1; j++) {
    float q_level = (float(q_indexes[j])/(n_simulations - 1)) * 100;
    s << ",\\" << endl << "\"" << fsource << "\" using 1:" << 2 + j << " title \"" << q_level << "% quantile\"with lines";
  }
  s << endl;
  write_to_file(f, s.str());
  f.close();
}





//k-means with trends
void gnuplot_script_kmeans(
			   string prefix,
			   string fsource,
			   string title,
			   double time_limit,
			   int n_clusters,
			   double time_delta,
			   unsigned int n_simulations
			   )
{
  double point_size = (double)POINT_SIZE / n_simulations;
  stringstream gnuplot_base;
  gnuplot_init(gnuplot_base, time_limit);
  stringstream r_gnuplot;
  r_gnuplot << gnuplot_base.str();
  r_gnuplot << "set ylabel \"K-means clusters with trends (" << title << ")\"" << endl;
  r_gnuplot << "set pointsize " << point_size << endl;
  //file
  string out_fname = prefix + "." + GNUPLOT_SCRIPT_EXTENSION;
  ofstream f(out_fname.c_str());
  if(!f.is_open()) {
    cerr << "Could not open file: " << out_fname << endl;
    exit(1);
  }
  //script
  stringstream s;
  s << r_gnuplot.str()
    << "set output \"" << prefix << "." << GNUPLOT_OUTPUT_EXTENSION << "\"" << endl
    //store time-delta
    << "td=" << time_delta << endl
    //centers
    //<< "plot \"" << fsource << "\" using 1:2:4 title \"cluster 1\" with points lt 0.1 lc rgb \"dark-gray\" pt 7 ps variable"
    << "plot \"" << fsource << "\" using 1:2:4 notitle with points lt 0.1 lc rgb \"dark-gray\" pt 7 ps variable"
    << ",\\" << endl
    << "\"" << fsource << "\" using 1:2:(td):($3-$2) notitle with vectors head filled lt 1";
  for(int j=1; j<n_clusters; j++) {
    s << ",\\" << endl
      //circle
      << "\"" << fsource << "\" using 1:" << 2 + 3*j << ":" << 2 + 3*j + 2
      //<< " title \"cluster " << j + 1 << "\" with points lt 0.1 lc rgb \"dark-gray\" pt 7 ps variable"
      << " notitle with points lt 0.1 lc rgb \"dark-gray\" pt 7 ps variable"
      << ",\\" << endl
      //trend
      << "\"" << fsource << "\" using 1:" << 2 + 3*j << ":(td):($" << 2 + 3*j + 1 << "-$" << 2 + 3*j << ")"
      << " notitle with vectors head filled lt " << j+1;
  }
  s << endl;
  write_to_file(f, s.str());
  f.close();
}





//qt with trends (init)
void gnuplot_script_qt_init(
			    ofstream **of,
			    string prefix,
			    string fsource,
			    string title,
			    double time_limit,
			    double time_delta,
			    unsigned int n_simulations
			    )
{
  double point_size = (double)POINT_SIZE / n_simulations;
  stringstream gnuplot_base;
  gnuplot_init(gnuplot_base, time_limit);
  stringstream r_gnuplot;
  r_gnuplot << gnuplot_base.str();
  r_gnuplot << "set ylabel \"QT clusters with trends (" << title << ")\"" << endl;
  r_gnuplot << "set pointsize " << point_size << endl;
  //file
  string out_fname = prefix + "." + GNUPLOT_SCRIPT_EXTENSION;
  *of = new ofstream(out_fname.c_str());
  if(!(*of) || !((**of).is_open())) {
    cerr << "Could not open file: " << out_fname << endl;
    exit(1);
  }
  ofstream &f(**of);
  //script
  stringstream s;
  s << r_gnuplot.str()
    << "set output \"" << prefix << "." << GNUPLOT_OUTPUT_EXTENSION << "\"" << endl
    //store time-delta
    << "td=" << time_delta << endl;
  write_to_file(f, s.str());
}

//qt with trends (finalize)
void gnuplot_script_qt_finalize(ofstream *of, string fsource, int n_clusters) {
  stringstream s;
  //centers
  //<< "plot \"" << fsource << "\" using 1:2:4 title \"cluster 1\" with points lt 0.1 lc rgb \"dark-gray\" pt 7 ps variable"
  s << "plot \"" << fsource << "\" using 1:2:4 notitle with points lt 0.1 lc rgb \"dark-gray\" pt 7 ps variable"
    << ",\\" << endl
    << "\"" << fsource << "\" using 1:2:(td):($3-$2) notitle with vectors head filled lt 1";
  for(int j=1; j<n_clusters; j++) {
    s << ",\\" << endl
      //circle
      << "\"" << fsource << "\" using 1:" << 2 + 3*j << ":" << 2 + 3*j + 2
      //<< " title \"cluster " << j + 1 << "\" with points lt 0.1 lc rgb \"dark-gray\" pt 7 ps variable"
      << " notitle with points lt 0.1 lc rgb \"dark-gray\" pt 7 ps variable"
      << ",\\" << endl
      //trend
      << "\"" << fsource << "\" using 1:" << 2 + 3*j << ":(td):($" << 2 + 3*j + 1 << "-$" << 2 + 3*j << ")"
      << " notitle with vectors head filled lt " << j+1;
  }
  s << endl;
  ofstream &f(*of);
  write_to_file(f, s.str());
  f.close();
  delete of;
}





//peak frequency
void gnuplot_script_peaks(string prefix, string fsource, string title, double time_limit) {
  stringstream gnuplot_base;
  gnuplot_init(gnuplot_base, time_limit);
  stringstream r_gnuplot;
  r_gnuplot << gnuplot_base.str();
  r_gnuplot << "set ylabel \"peaks (" << title << ")\"" << endl;
  //r_gnuplot << "set pointsize " << point_size << endl;
  //file
  string out_fname = prefix + "." + GNUPLOT_SCRIPT_EXTENSION;
  ofstream f(out_fname.c_str());
  if(!f.is_open()) {
    cerr << "Could not open file: " << out_fname << endl;
    exit(1);
  }
  //script
  stringstream s;
  s << r_gnuplot.str();
  s << "set output \"" << prefix << "." << GNUPLOT_OUTPUT_EXTENSION << "\"" << endl;
  //centers
  s << "plot \"" << fsource << "\" using 1:2 title \"(avg) peak frequency\" with points";
  s << endl;
  write_to_file(f, s.str());
  f.close();
}





void gnuplot_script_all(string prefix,
			string kmeans_source,
			string raw_source,
			double time_limit,
			unsigned int monitor_id,
			unsigned int n_simulations,
			unsigned int n_clusters
			) {
  stringstream gnuplot_base;
  gnuplot_init(gnuplot_base, time_limit);
  stringstream r_gnuplot;
  r_gnuplot << gnuplot_base.str();
  r_gnuplot << "set ylabel \"curves and k-means (monitor " << monitor_id << ")\"" << endl;
  //file
  string out_fname = prefix + "." + GNUPLOT_SCRIPT_EXTENSION;
  ofstream f(out_fname.c_str());
  if(!f.is_open()) {
    cerr << "Could not open file: " << out_fname << endl;
    exit(1);
  }
  //script
  stringstream s;
  s << r_gnuplot.str();
  s << "set output \"" << prefix << "." << GNUPLOT_OUTPUT_EXTENSION << "\"" << endl;
  //curves
  s << "plot \"" << raw_source << "0\" using 1:" << 2 + monitor_id << " title \"curve 0\" with lines";
  for(unsigned int i=1; i<n_simulations; ++i)
    s << ",\\\n\"" << raw_source << i << "\" using 1:" << 2 + monitor_id << " title \"curve " << i << "\" with lines";
  //kmeans
  for(unsigned int j=0; j<n_clusters; j++) {
    s << ",\\" << endl;
    s << "\"" << kmeans_source << "\" using 1:" << 2 + 2*j << ":" << 2 + 2*j + 1;
    s << " title \"sizes of cluster " << j + 1 << "\" with yerrorbars";
    s << ",\\" << endl;
    s << "\"" << kmeans_source << "\" using 1:" << 2 + 2*j;
    s << " title \"cluster " << j + 1 << "\""<< " with lines";
  }
  s << endl;
  write_to_file(f, s.str());
  f.close();
}
