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

#ifndef OUTPUT
#define OUTPUT
#include <fstream>
#include <string>
#include <vector>
using namespace std;
#include "definitions.h"

void write_to_file(ofstream &, string);
void write_sample_to_file(ofstream &, double, vector<multiplicityType> &);
//void write_reduction_to_file(ofstream &, double, vector<double> &);
#endif
