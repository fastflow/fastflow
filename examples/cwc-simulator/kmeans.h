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

#ifndef _KMEANS_H_
#define _KMEANS_H_

#include "definitions.h"
#include <vector>
#include <valarray>
using namespace std;

unsigned int find_nearest_cluster(point_t &, vector<centroid_t> &, unsigned int);

void kmeans(
	    vector<valarray<double> > &curves, //src: windows (one per simulation)
	    vector<centroid_t> &centroids, //dst: centroids
	    vector<point_t> &points, //dst: clustered points
	    double time_win, //window sampling time step
	    unsigned int window_size, //window length
	    unsigned int degree, //interpolating polynomial degree
	    double prediction_factor, //prediction factor
	    unsigned int fw_left, //window left-limit
	    unsigned int fw_right //window right-limit
	    );
#endif
