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
#include "MatchSet.h"
#ifdef HYBRID
#include "ode.h"
#endif
#include "Gillespie.hpp"
#ifdef GRAIN
#include <ff/cycle.h>
using namespace ff;
#endif

//class Gillespie;

#include <iostream>
#include <fstream>
using namespace std;

class Simulation {
  friend ostream& operator<<(ostream &, const Simulation &);

 public:
  Simulation(int id, Model model, int seed
#ifdef HYBRID
	     , double rc,
	     multiplicityType pc,
	     double
#endif
	     );
  ~Simulation();

  double get_time();
  int get_id();

  double next_simulation_tau();
  void compute_updates();
  void compute_ode_delta(double);

  void step(double);
  void ode(double);

  void freeze(double);
  double restore();

  void set_stall();
  bool get_stall();

#ifdef GRAIN
  double get_cycle_ticks();
  unsigned int get_steps();
#endif

  vector<multiplicityType> *monitor();

#ifdef LOG
  ofstream &get_logfile();
#endif

 private:
  MatchSet ms;
  int id;
  double time; //current time
  Model model; //current state
  double freeze_tau; //next tau
  bool stall;
#ifdef HYBRID
  double time_ode; //ode time
  double rate_cutoff;
  multiplicityType population_cutoff;
  ode_delta_t ode_delta;
#endif
  Gillespie *gillespie;
  updates_t updates;
#ifdef LOG
  ofstream *logfile;
#endif
#if defined(LOG) || defined(GRAIN)
  unsigned long steps;
#endif
#ifdef GRAIN
  ticks svc_ticks;
  double cycle_ticks;
#endif
};
#endif
