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

#include "statistics.h"
#include <boost/math/distributions/students_t.hpp>
#include <cmath>
using namespace boost::math;

//#include <iostream>
//using namespace std;

double sample_mean_sub(multiplicityType *sample, int lw, int up) {
  double acc = 0;
  for(int i=lw; i<=up; i++)
    acc += sample[i];
  return acc / (up - lw + 1);
}

double sample_variance_sub(double sample_mean, multiplicityType *sample, int lw, int up) {
  double sv = 0;
  for(int i=lw; i<=up; i++) {
    double error = sample[i] - sample_mean;
    sv += (error * error);
  }
  return (1 / double(up - lw)) * sv; //size - 1 = up - lw + 1 - 1
}

double sample_mean(multiplicityType *sample, int size) {
  return sample_mean_sub(sample, 0, size - 1);
}

double sample_variance(double sample_mean, multiplicityType *sample, int size) {
  return sample_variance_sub(sample_mean, sample, 0, size - 1);
}

double sample_standard_deviation(double sample_variance) {
  return sqrt(sample_variance);
}

void grubb_reductions(multiplicityType *sample, int n, double grubb, double &mean, double &variance) {
  //test on:
  //h0: no outliers
  //ha: at least one outiler
  int limits[2], to_change, into;
  limits[0] = 0; //lw
  limits[1] = n - 1; //up
  double g_min, g_max, g, t2_q, ssd;
  mean = 0;
  variance = 0;
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
}

double confidence_interval(double sample_standard_deviation, int size, double significance) {
  //using student's T
  students_t distribution(max(size - 1, 1));
  double t = quantile(complement(distribution, significance / 2));
  //cout << "t-quantile for confidence " << confidence/2 << ": " << t << endl;
  return t * sample_standard_deviation / sqrt(double(size));
}

multiplicityType sample_min(multiplicityType *sample, int size) {
  multiplicityType x = sample[0];
  for(int i=1; i<size; i++)
    if(sample[i] < x)
      x = sample[i];
  return x;
}

multiplicityType sample_max(multiplicityType *sample, int size) {
  multiplicityType x = sample[0];
  for(int i=1; i<size; i++)
    if(sample[i] > x)
      x = sample[i];
  return x;
}

//void quantiles(vector<multiplicityType> &sample, vector<int> &q_indexes, vector<unsigned int> &q) {
void quantiles(multiplicityType *sample, vector<int> &q_indexes, multiplicityType *q) {
  //pre: sample is sorted
  for(unsigned int i=0; i<q_indexes.size(); i++)
    q[i] = (sample[q_indexes[i]]);
}

vector<int> linspace_int(int size, int points) {
  vector<int> res(points, 0);
  float step = float(size -1) / (points - 1);
  for(int i=0; i < points - 1; i++)
    res[i] = int(round(i * step));
  res[points - 1] = size -1;
  return res;
}
