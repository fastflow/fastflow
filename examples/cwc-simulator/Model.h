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

#ifndef MODEL
#define MODEL
#include "definitions.h"
#include <limits>
#include <string>
#include <sstream>
#include <vector>
#ifdef LOG
#include <ostream>
#endif

#include "Compartment.h"
class Monitor;
class Rule;
class Species;
class PCompartment;
//class Compartment;

typedef vector<string> reverse_symbol_table;

/**
   CWC model
*/
class Model {
 public:
  /**
     a brief description of the model
  */
  string title;
	
  /**
     adressing table over the alphabet
  */
  reverse_symbol_table reverse_atom_table;

  /**
     the ruleset
  */
  vector<Rule *> rules;

  /**
     size of the ODE system
  */
  unsigned int ode_size;
	
  /**
     the initial term
  */
  Compartment &term;
	
  /**
     patterns to be monitored
  */
  vector<Monitor *> monitors;

  /**
     contructor
  */
  Model(
	string &description,
	reverse_symbol_table &,
	vector<Rule *> &ruleset,
	unsigned int ode_size,
	Species &init_species,
	vector<Compartment *> &init_compartments,
	vector<Monitor *> &monitors
	);

  Model(Model &);
  ~Model();

  /**
     produces a string which resumes the model to be simulated
     @param time_limit the time at which the simulation will halt
     @return the resuming string
  */
  string header(double);

  /**
     get the multiplicity of each pattern-monitor
     @return a vector of the multiplicities (one for each monitor)
  */
  sample_t *monitor(
#ifdef LOG
		    std::ostream &
#endif
);

  /**
     get the number of the species
     @return the number of the species
   */
  unsigned int n_species();

  friend std::ostream& operator<<(std::ostream &, const Model &);

 private:
    multiplicityType count(PCompartment &, Compartment &);
};
#endif
