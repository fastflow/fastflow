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

#include "Simulation.h"
#include "Gillespie.hpp"
#include <sstream>
#include <fstream>
#include "utils.h"
using namespace std;

Simulation::Simulation(int id, Model m, int seed
#ifdef HYBRID
		       , double rc,
		       multiplicityType pc,
		       double sampling_rate
#endif
	     ) :
  id(id), time(0), model(m)
#ifdef GRAIN
  , svc_ticks(0), cycle_ticks(0)
#endif
#ifdef HYBRID
  , time_ode(0),
  rate_cutoff(rc),
  population_cutoff(pc)
#endif
{

#ifdef LOG
  stringstream logfname;
  logfname << "log_sim" << id;
  logfile = new ofstream(logfname.str().c_str());
  if(!logfile->is_open()) {
    delete logfile;
    cerr << "Could not open file: " << ("log_sim") + id << endl;
    exit(1);
  }

#endif
  ms.init(model

#ifdef LOG
	  , logfile
#endif
	  );

#ifdef LOG
  MYNEW(gillespie, Gillespie, seed, *logfile);
#else
	   MYNEW(gillespie, Gillespie, seed);
#endif

#ifdef HYBRID
  ode_delta.t_sampling = sampling_rate;
#endif

  stall = false;
  if((freeze_tau = next_simulation_tau()) != -1) {
    compute_updates();
#ifdef HYBRID
    compute_ode_delta(freeze_tau);
#endif
  }
  else
    set_stall();

#ifdef LOG
  steps = 0;
  *logfile << "[0] " << model.term << endl;
#endif
}

Simulation::~Simulation() {
  delete gillespie;
  ms.clear();
#ifdef LOG
  *logfile << steps << " steps" << endl;
  if(logfile) {
    logfile->close();
    delete logfile;
  }
#endif
}

double Simulation::get_time() {
  return time;
}

int Simulation::get_id() {
  return id;
}

#ifdef LOG
ofstream &Simulation::get_logfile() {
  return *logfile;
}
#endif

void Simulation::set_stall() {
  stall = true;
}

bool Simulation::get_stall() {
  return stall;
}



/*
  compute the time offset at which the next reaction will occur
*/
double Simulation::next_simulation_tau() {

#ifdef GRAIN
  ticks s = getticks();
#endif

  //pattern matching to fill the matchset
  ms.match(model, time);

#ifdef GRAIN
  svc_ticks += getticks() - s;
#endif

#ifndef HYBRID
#ifdef SIMLOG
  *logfile << ms << endl;
#endif
#endif

  if(ms.rate_sum > 0) {
#ifdef HYBRID
    ms.split(rate_cutoff, population_cutoff, model.ode_size);
#ifdef SIMLOG
    *logfile << ms << endl;
#endif
    if(ms.rate_sum > 0)
#endif
      return gillespie->tau(ms);
#ifdef HYBRID
    else {
      //retry
      ms.ode_only = true;
      return gillespie->tau_retry(ms);
    }
#endif
  }

  else
    //full stall
    return -1;
}



/*
  compute the updates for the next reaction
 */
void Simulation::compute_updates() {
#ifdef GRAIN
  ticks s = getticks();
#endif
  gillespie->compute_updates(updates, ms);
#ifdef GRAIN
  svc_ticks += getticks() - s;
#endif
}

#ifdef HYBRID
void Simulation::compute_ode_delta(double tau) {
  ode_delta.t_start = time;
  ode_delta.t_end = time + tau;
  gillespie->compute_ode_delta(ode_delta, ms, updates, tau, model.n_species());
}
#endif

void Simulation::step(double tau) {
#ifdef GRAIN
  ticks s = getticks();
#endif

  gillespie->update(ms, updates
#ifdef HYBRID
		    , ode_delta
#endif
		    ); //TODO: check
  time += tau;

#ifdef HYBRID
  //align ode time with sim. time
  time_ode = time;
#endif

#ifdef LOG
  *logfile << "[" << time << "]: " << (model.term) << endl;
#endif

#if defined(LOG) || defined(GRAIN)
  ++steps;
#endif
  ms.clear();

#ifdef GRAIN
  svc_ticks += getticks() - s;
  cycle_ticks = (double)svc_ticks / steps;
#endif
}

#ifdef HYBRID
void Simulation::ode(double t_target) {
  gillespie->ode(ode_delta, t_target);
}
#endif

void Simulation::freeze(double t) {
  freeze_tau = t;
}

double Simulation::restore() {
  return freeze_tau;
}

#ifdef GRAIN
double Simulation::get_cycle_ticks() {
  return cycle_ticks;
}

unsigned int Simulation::get_steps() {
  return steps;
}
#endif

vector<multiplicityType> * Simulation::monitor() {
  return model.monitor(
#ifdef LOG
		       *logfile
#endif
);
}

ostream& operator<<(ostream &os, const Simulation &s) {
  return os << "Simulation " << s.id << ". [ " << s.time << "]: " << &(s.model.term);
}
