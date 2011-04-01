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

#ifndef ODE_H
#define ODE_H

#include "definitions.h"
#include "Species.h"

#include <limits>
using namespace std;

#define RATE_CUTOFF_INFINITE -1 //not sure about comparation of doubles
#define POPULATION_CUTOFF_INFINITE numeric_limits<multiplicityType>::max()

void limited_ode(vector<double> &k, Species &x, double tau);
#endif
