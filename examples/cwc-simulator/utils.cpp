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

#include "utils.h"
#include <vector>
using namespace std;

/* Based on round.awk by Arnold Robbins, Public Domain */ 
int cwc_utils::my_round (double x) {
  int i = (int) x; //floor
  if (x >= 0.0)
    return i + ((x-i) >= 0.5);
  else
    return i - (-x+i >= 0.5);
}

vector<int> cwc_utils::linspace_int(int size, int points) {
  vector<int> res(points, 0);
  double step = double(size -1) / (points - 1);
  for(int i=0; i < points - 1; i++)
    res[i] = int(my_round(i * step));
  res[points - 1] = size -1;
  return res;
}
