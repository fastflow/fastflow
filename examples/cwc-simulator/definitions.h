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

#ifndef _CWC_DEFINITIONS_H_
#define _CWC_DEFINITIONS_H_
#include <string>
#include <vector>
using namespace std;

typedef /*unsigned*/ long long multiplicityType;
typedef string variable_symbol;
typedef vector<multiplicityType> sample_t; //samples (one for each monitor)

typedef unsigned int type_adress;
typedef unsigned int symbol_adress;
typedef unsigned int variable_adress;

//clustered point
typedef struct point {
  double position; //position in the trajectory
  double position_filtered; //filtered position
  double prediction; //predicted position
  unsigned int membership; //cluster membership

point() : position(0.0), position_filtered(0.0), prediction(0.0), membership(0) {}
} point_t;

//k-means centroid
typedef struct centroid {
  double position;
  double prediction;
  unsigned int size;
} centroid_t;

#endif
