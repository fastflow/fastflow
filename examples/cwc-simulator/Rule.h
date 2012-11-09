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

#ifndef RULE
#define RULE
#include "OCompartment.h"
#include "PCompartment.h"
/*
#include "Compartment.h"
#include "OCompartment.h"
#include "PCompartment.h"
*/
#include <map>
#include <string>
#include <vector>
using namespace std;

class Compartment;
//class OCompartment;
//class PCompartment;
class Species;

typedef unsigned long long semanticsType;

/**
   CWC stochastic rule.
*/
class Rule {

 public:
  /**
     left hand side of the rule (CWC pattern)
  */
  PCompartment &lhs;
	
  /**
     right hand side of the rule (CWC open term)
  */
  OCompartment &rhs;

  /**
     sematics-type of the rule
   */
  semanticsType semantics;
	
  /**
     parameters for of the rule-sematics
  */
  vector<double> parameters;
	
  /**
     type of the rule
  */
  type_adress type;

  /**
     biochemical-format shape
  */
  bool is_biochemical;

  /**
     starting adress of the equation-parameters in the ODE system
  */
  unsigned int ode_index;

  /**
     species to be searched for in matching contexts
   */
  vector<symbol_adress> species_occs;
	
  /**
     constructor
  */
  Rule(
       Species &lhs_species,
       vector<PCompartment *> &lhs_compartments,
       Species &rhs_species,
       vector<OCompartment *> &rhs_compartments,
       semanticsType s,
       vector<double> p,
       vector<symbol_adress> *so,
       //variable_symbol &lhs_variable,
       //vector<variable_symbol *> &rhs_variables,
       variable_adress lhs_variable,
       vector<variable_adress> &rhs_variables,
       symbol_adress t
       );

  /**
     copy constructor
  */
  Rule(Rule &);

  ~Rule();

  friend std::ostream& operator<<(std::ostream &, const Rule &);
};
#endif
