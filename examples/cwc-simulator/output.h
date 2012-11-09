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

#ifndef _CWC_OUTPUT_H_
#define _CWC_OUTPUT_H_
#include "definitions.h"
//#include "kmeans.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
using namespace std;

void write_to_file(ofstream &out, string s);
void write_sample_to_file(ofstream &out, double time, vector<multiplicityType> &monitors);

template <class T>
void write_stat(ofstream *f, vector<T> &stat, double time) {
  stringstream s;
  s << time << "\t";
  for(unsigned int i=0; i<stat.size(); i++)
    s << stat[i] << "\t";
  string out_s = s.str();
  out_s.erase(out_s.length() - 1); //erase the last \t
  write_to_file(*f, out_s + "\n");
}

void write_stat_notzero(ofstream *f, vector<centroid_t> &stat, double time); //kmeans
void write_stat_qt(ofstream *f, vector<point_t> &stat, vector<list<int> > &card, double time); //QT
#endif
