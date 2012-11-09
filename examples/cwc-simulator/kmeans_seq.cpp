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

#include "kmeans.h"
#include "filter.h" //prediction
#include <limits>
using namespace std;

#define MAX_LOOPS 500


unsigned int find_nearest_cluster(point_t &point, vector<centroid_t> &centroids, unsigned int n_clusters) {
  double distance = (numeric_limits<double>::max)();
  unsigned int nearest = 0;
  double d_position, d_prediction;
  for(unsigned int i=0; i<n_clusters; ++i) {
    d_position = abs(point.position_filtered - centroids[i].position);
    d_prediction = abs(point.prediction - centroids[i].prediction);
    double d_weighted = d_position / 3 + d_prediction * 2.0 / 3;
    if(d_weighted < distance) {
      distance = d_weighted;
      nearest = i;
    }
  }
  return nearest;
}

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
	    )
{
  unsigned int n_clusters = centroids.size();
  vector<centroid_t> old_centroids(centroids);

  //update (clustered) points with prediciton
  unsigned int n_simulations = points.size();
  Savgol_Prediction savgol(fw_left, fw_right, degree, prediction_factor * time_win, time_win);
  for(unsigned int i=0; i<n_simulations; ++i) {
    points[i].position_filtered = savgol.filt(curves[i]);
    points[i].position = curves[i][fw_left];
    points[i].prediction = savgol.prediction(curves[i]);
  }

  //compute clustering
  bool fix = false;
  unsigned loop = 0;
  double delta = 0.0;
  double threshold = 0.1;
  unsigned int old_membership, new_membership;
  while(!fix) {
    delta = 0.0;
    //reset centroids
    for(unsigned int i=0; i<n_clusters; ++i)
      centroids[i].position = centroids[i].prediction = centroids[i].size = 0;
    for(unsigned int i=0; i<n_simulations; ++i) {
      //find nearest cluster
      old_membership = points[i].membership;
      new_membership = find_nearest_cluster(points[i], old_centroids, n_clusters);
      //assign membership
      points[i].membership = new_membership;
      delta += (new_membership != old_membership);
      /*
	if(new_membership != old_membership) {
	delta += 1.0;
	cerr << "switch: {pos: " << points[i].position << ", pred: " << points[i].prediction << "}"
	<< "from {pos: " << old_centroids[old_membership].position  << ", pred: " << old_centroids[old_membership].prediction << "} "
	<< "to {pos: " << old_centroids[new_membership].position << ", pred: " << old_centroids[new_membership].prediction << "}" << endl;
	}
      */
      //update cluster
      centroids[new_membership].size += 1;
      centroids[new_membership].position += points[i].position;
      centroids[new_membership].prediction += points[i].prediction;
    }
    //adjust centroids (divide by size)
    for(unsigned int i=0; i<n_clusters; ++i) {
      if(centroids[i].size > 0) {
	//non-empty cluster
	centroids[i].position /= centroids[i].size;
	centroids[i].prediction /= centroids[i].size;
      }
      //backup clusters
      old_centroids[i].position = centroids[i].position;
      old_centroids[i].prediction = centroids[i].prediction;
      old_centroids[i].size = centroids[i].size;
    }
    //update convergence measure
    fix = (delta / n_simulations) < threshold || ++loop > MAX_LOOPS;
  }
  /*
#ifdef LOG
  cerr << "clustering state:" << endl;
  for(unsigned int i=0; i<n_clusters; ++i)
    cerr << "centroid " << i << ": " << centroids[i].position << "(p: " << centroids[i].prediction << ")" << endl;
  for(unsigned int i=0; i<n_simulations; ++i)
    cerr << "point " << i << ": {membership = " << points[i].membership << ", position = " << points[i].position << ", prediction = " << points[i].prediction << "}" << endl;
#endif
  */
}

