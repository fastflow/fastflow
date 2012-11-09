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

#ifndef PCOMPARTMENT
#define PCOMPARTMENT
#include "definitions.h"
//#include "PSpecies.h"
//#include "Compartment.h"
//#include "SubMatchSet.h"
//#include "utils.h"
#include <vector>
#include <string>
using namespace std;

class SubMatchSet;
class Species;
class PSpecies;
class Compartment;

//cache of submatchsets
typedef struct sms_cache_item {
  SubMatchSet *sms;
  bool stored;
} sms_cache_item_t;



/**
   Compartment of a CWC pattern.
*/
class PCompartment {
	
 public:
  /**
     wrap of the compartment
  */
  PSpecies *wrap;

  /**
     content-species of the compartment
  */
  PSpecies *content_species;

  /**
     content-compartments of the compartment
  */
  vector<PCompartment *> *content_compartments;

  /**
     type of the compartment (adress)
  */
  symbol_adress type;

  /**
     wrap variable
  */
  //variable_symbol wrap_variable_symbol;
  variable_adress wrap_variable_symbol;
	
  /**
     content variable
  */
  //variable_symbol content_variable_symbol;
  variable_adress content_variable_symbol;

  /**
     constructor
  */
  PCompartment (
		Species *wrap,
		Species *cs,
		vector<PCompartment *> *cc,
		//variable_symbol &wrap_variable,
		//variable_symbol &content_variable,
		variable_adress wrap_variable,
		variable_adress content_variable,
		symbol_adress t
		);

  /**
     dummy constructor
  */
  //PCompartment ();

  /**
     top-level constructor
  */
  PCompartment (
		Species *cs,
		vector<PCompartment *> *cc,
		//variable_symbol &content_variable,
		variable_adress content_variable,
		symbol_adress t
		);

  /**
     copy constructor
  */
  PCompartment (PCompartment &);

  ~PCompartment();

  /**
     matching against a subject simple term.
     It also triggers submatchings on nested compartments.
     @param t the subject compartment
     @param sms the submatchset on which triggered submatchings are called
     @return the number of matches
  */
  multiplicityType match(Compartment &c, SubMatchSet *sms);

  /**
     matching (of a biochemical pattern) against a subject simple term.
     @param t the subject compartment
     @return the number of matches and the rarest multiplicity of a species
  */
  pair<multiplicityType, multiplicityType> match_biochemical(Compartment &c);

  /**
     counting against a subject compartment.
     @param c the subject compartment
     @return the number of matches
  */
  multiplicityType count(Compartment &c);

  friend ostream& operator<<(ostream &, PCompartment &);

 private:
  unsigned int n_content_compartments;
  sms_cache_item_t **sms_cache;
  unsigned int n_subjects_last;

};
#endif
