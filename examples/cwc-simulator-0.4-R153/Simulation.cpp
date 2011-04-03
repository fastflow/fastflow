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
#include "random.h"
#include "MatchSet.h"
#include "Gillespie.h"

//#include <iostream>
//using namespace std;

//Simulation::Simulation(int id, Model model, u01_vg_type vg
Simulation::Simulation(int id, Model model, int seed
#ifdef HYBRID
	     , double rc,
	     multiplicityType pc
#endif
	     ) :
  id(id), time(0), model(model)
  //random_generator(vg)
#ifdef HYBRID
  , rate_cutoff(rc),
  population_cutoff(pc)
#endif
{
  rng_type sim_rng(seed);
  u01_gm_type sim_gm;
  random_generator = new u01_vg_type(sim_rng, sim_gm);
}

Simulation::~Simulation() {
  delete random_generator;
}

double Simulation::get_time() {
  return time;
}

int Simulation::get_id() {
  return id;
}



//Pure stochastic semantics
#ifndef HYBRID
double Simulation::step() {
  //match
  MatchSet ms;
  ms.match(model, time);

  if(ms.rate_sum > 0) {
#ifdef DEBUG_SIMULATION
    if(id == 0) {
      cerr << endl << "(" << time << ") term: " << model.term << endl;
      cerr << "matchset:" << endl << ms << "-" << endl;
    }
#endif

    stochastic_gillespie(ms, time, *random_generator);
    return time;
  }

  //else
  //full stall
  return -1;
}





//Hybrid semantics
#else
double Simulation::step() {
  //match
  MatchSet ms;
  ms.match(model, time);

  if(ms.rate_sum > 0) {
    //split
#ifdef DEBUG_SIMULATION
    if(id == 0) {
      cerr << endl << "(" << time << ") term: " << model.term << endl;
      cerr << "pre-split:" << endl << ms << "-" << endl;
    }
#endif
    ms.split(rate_cutoff, population_cutoff, model.ode_size); //comment to test with pure stochastic semantics
#ifdef DEBUG_SIMULATION
    if(id == 0) {
      cerr << "post-split:" << endl << ms << "-" << endl;
    }
#endif

    if(ms.rate_sum > 0) {
      //hybrid step
      //cerr << "hybrid step; rate total: " << ms.rate_sum << endl;
      hybrid_gillespie(ms, time, *random_generator);
      return time;
    }

    else {
      //stochastic stall
      //cerr << "no stochastic rules: will retry with ode" << endl;
      //cerr << "should not be here" << endl; //uncomment to test with pure stochastic semantics
      retry(ms, time, *random_generator);
      return time;
    }
  }

  //else
  //full stall
  return -1;
}
#endif



vector<multiplicityType> * Simulation::monitor() {
  //cerr << *this << endl;
  return model.monitor();
}

ostream& operator<<(ostream &os, const Simulation &s) {
  return os << "Simulation " << s.id << ". [ " << s.time << "]: " << s.model.term;
}
