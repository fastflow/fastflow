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

#ifndef PROGRESSBARS_HPP
#define PROGRESSBARS_HPP
#include "utils.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
using namespace std;
using namespace cwc_utils;
#define MAX_OUTPUT_BARS 10

class ProgressBars {
public:
  ProgressBars(int n) {
    //init bars
    for(int i=0; i<n; i++)
      bars.push_back(0);
    //get indexes
    if(n <= MAX_OUTPUT_BARS) {
      for(int j=0; j<n; j++)
	indexes.push_back(j);
      ni = n;
    }
    else {
      indexes = linspace_int(n, MAX_OUTPUT_BARS);
      ni = MAX_OUTPUT_BARS;
    }
    //print lines
    for(int i = 0; i < ni; i++) {
      cout.width(5);
      cout << (indexes[i] + 1) << "|";
    }
    cout << endl;
    for(int i = 0; i < ni; i++) {
      cout << "------";
    }
    cout << endl;
  }

  void set(int i, double percent) {
    bars[i] = percent;
    stringstream res;
    for(int j = 0; j < ni; j++) {
      res.width(4);
      res << int(bars[indexes[j]]) << "%|";
    }
    cout << "\r" << res.str() << flush;
  }

private:
  vector<double> bars;
  vector<int> indexes;
  int ni;
};
#endif
