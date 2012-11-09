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

#ifndef OCOMPARTMENT
#define OCOMPARTMENT

//#include "definitions.h"
//#include "Species.h"
#include "Compartment.h"
#include "Instantiation.h"
#include <vector>
#include <string>
using namespace std;

/**
   compartment of a CWC open term
*/
class OCompartment {
 public:
  /**
     wrap of the compartment
  */
  Species *wrap;

  /**
     content-species of the compartment
  */
  Species *content_species;

  /**
     content-compartments of the compartment
  */
  vector<OCompartment *> *content_compartments;

  /**
     type of the compartment (adress)
  */
  symbol_adress type;

  /**
     wrap variables
  */
  //vector<variable_symbol *> &wrap_variables_symbols;
  vector<variable_adress> wrap_variables_symbols;
  /**
     content variables
  */
  //vector<variable_symbol *> &content_variables_symbols;
  vector<variable_adress> content_variables_symbols;

  /**
     constructor
  */
  OCompartment (
		Species *wrap,
		Species *cs,
		vector<OCompartment *> *cc,
		//vector<variable_symbol *> &wrap_variables,
		//vector<variable_symbol *> &content_variables,
		vector<variable_adress> wrap_variables,
		vector<variable_adress> content_variables,
		symbol_adress t
		);

  /**
     top-level constructor
  */
  OCompartment(
	       Species *cs,
	       vector<OCompartment *> *cc,
	       //vector<variable_symbol *> &content_variables,
	       vector<variable_adress> content_variables,
	       symbol_adress t
	       );

  /**
     copy constructor
  */
  OCompartment (OCompartment &);
	
  ~OCompartment();

  /**
     build the ground compartment by an instantiation
     @param sigma the instantiation
     @return the ground compartment
  */
  Compartment *ground(instantiationType &sigma);

  friend ostream& operator<<(ostream &, OCompartment &);
};
#endif
