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
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
using namespace std;

void write_to_file(ofstream &out, string s) {
  out << s << flush;
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

//kmeans
//columns: time, c1 centroid, c1 predicted centroid, c1 size, ... 
void write_stat_notzero(ofstream *f, vector<centroid_t> &stat, double time) {
  stringstream s;
  s << time << "\t";
  for(unsigned int i=0; i<stat.size(); i++) {
    if (stat[i].size > 0) {
      //not empty cluster
      s << stat[i].position << "\t"
	<< stat[i].prediction << "\t"
	<< stat[i].size << "\t";
    }
    else {
      //join the nearest not empty cluster
      s << stat[(i-1)%stat.size()].position << "\t"
	<< stat[(i-1)%stat.size()].prediction << "\t"
	<< stat[i].size << "\t";
    }
  }
  string out_s = s.str();
  out_s.erase(out_s.length() - 1); //erase the last \t
  write_to_file(*f, out_s + "\n");
}

//QT
//columns: time, c1 center, c1 size, ...
void write_stat_qt(ofstream *f, vector<point_t> &stat, vector<list<int> > &card, double time) {
  stringstream s;
  s << time << "\t";
  for(unsigned int i=0; i<stat.size(); i++)
    s << stat[i].position << "\t" << stat[i].prediction << "\t" << card[i].size() <<"\t";
  string out_s = s.str();
  out_s.erase(out_s.length() - 1); //erase the last \t
  write_to_file(*f, out_s + "\n");
}
