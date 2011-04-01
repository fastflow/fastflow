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

#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>
#include "definitions.h"
using namespace std;

#define N_PINS 5

void grubb_reductions(multiplicityType *sample, int n, double significance, double &mean, double &variance);
void reductions(multiplicityType *sample, int n, double significance, double &mean, double &variance);

vector<int> linspace_int(int size, int points);

//void quantiles(vector<multiplicityType> &sample, vector<int> &q_indexes, vector<unsigned int> &q);
void quantiles(multiplicityType *sample, vector<int> &q_indexes, multiplicityType *q);

#endif
