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

#ifndef SIMULATION
#define SIMULATION

#include "Model.h"
#include "random.h"
#ifdef HYBRID
#include "ode.h"
#endif
//#include "ff_accel.hpp"

#include <ostream>
using namespace std;

class Simulation {
  friend ostream& operator<<(ostream &, const Simulation &);

 public:
  //Simulation(int id, Model model, u01_vg_type vg
  Simulation(int id, Model model, int seed
#ifdef HYBRID
	     , double rc,
	     multiplicityType pc
#endif
	     );
  ~Simulation();

  double get_time();
  int get_id();
  double step();
  vector<multiplicityType> *monitor();

 private:
  int id;
  double time; //current time
  Model model; //curent state
  u01_vg_type *random_generator; //random number generator (uniform in 0-1)
#ifdef HYBRID
  double rate_cutoff;
  multiplicityType population_cutoff;
#endif
};
#endif
