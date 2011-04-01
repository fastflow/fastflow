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

#include "output.h"
#include <sstream>

void write_to_file(ofstream &out, string s) {
  out << s;
}

void write_sample_to_file(ofstream &out, double time, vector<multiplicityType> &monitors) {
  stringstream row;
  row << time << "\t";
  for(unsigned int i=0; i<monitors.size(); i++) {
    row << monitors[i] << "\t";
  }
  string res = row.str();
  res.erase(res.length() - 1); //erase the last \t
  out << res << endl;
}

/*
void write_reduction_to_file(ofstream &out, double time, vector<double> &reduction) {
  stringstream row;
  row << time << "\t";
  for(unsigned int i=0; i<reduction.size(); i++) {
    row << reduction[i] << "\t";
  }
  string res = row.str();
  res.erase(res.length() - 1); //erase the last \t
  out << res << endl;
}
*/
