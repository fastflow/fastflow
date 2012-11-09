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

#ifndef _CWC_GNUPLOT_H_
#define _CWC_GNUPLOT_H_
#include <string>
#include <vector>
using namespace std;

void gnuplot_script_reductions(string prefix, string fsource, string title, double time_limit, double grubb);

void gnuplot_script_classifications(string prefix, string fsource, string title, double time_limit, int n_simulations, vector<int> &q_indexes);

//void gnuplot_script_kmeans(string prefix, string fsource, string title, double time_limit, int n_clusters, double point_size);
void gnuplot_script_kmeans(string prefix, string fsource, string title, double time_limit, int n_clusters, double time_delta, unsigned int n_simulations); //with trends
void gnuplot_script_all(string prefix, string kmeans_source, string raw_source, double time_limit, unsigned int monitor_id, unsigned int n_simulations, unsigned int n_clusters);

void gnuplot_script_qt_init(ofstream **of, string prefix, string fsource, string title, double time_limit, double time_delta, unsigned int n_simulations);
void gnuplot_script_qt_finalize(ofstream *of, string fsource, int n_clusters);

void gnuplot_script_peaks(string prefix, string fsource, string title, double time_limit);
#endif
