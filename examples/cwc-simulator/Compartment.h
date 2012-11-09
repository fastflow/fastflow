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

#ifndef _CWC_COMPARTMENT_H_
#define _CWC_COMPARTMENT_H_

//#include "definitions.h"
#include "Species.h"
#include <vector>
#include <ostream>
using namespace std;

#define TOP_LEVEL 1
#define ANY_TYPE 0
#define VARIABLE_ADRESS_RESERVED 0

/**
   CWC compartment.
*/
class Compartment {

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
  vector<Compartment *> *content_compartments;

  /**
     type of the compartment (adress)
  */
  symbol_adress type;

  /**
     constructor
  */
  Compartment(Species *wrap, Species *cs, vector<Compartment *> *cc, symbol_adress t);

  /**
     top-level constructor
  */
  Compartment(Species *cs, vector<Compartment *> *cc, symbol_adress t = TOP_LEVEL);

  /**
     copy constructor
  */
  Compartment(Compartment &copy);
  ~Compartment();

  /**
     partially subtract a compartment.
     This subtraction regards the wrap and the species of the content;
     it doesn't affect nested compartments at all.
     @param c the compartment to be subtracted
     (it has to be superimposable to this in wrap and content-bags)
     @return the residual multiplicity (0 if nothing has been deleted)
  */
  void update_delete(Compartment &c);

  /**
     replace the content of the compartment.
     @param new_species the new content-species
     @param new_compartments the new content-compartments
  */
  void replace_content(Species *new_species, vector<Compartment *> *new_compartments);

  /**
     add species to the content
     @param delta the species to be added
   */
  void add_content_species(Species &delta);

  /**
     compute the difference between two compartments
     @param object the object-compartment
     @param the target diff
   */
  void trunked_content(Compartment &object, Species &diff);

  friend ostream& operator<<(ostream &, Compartment &);

 private:
  void delete_content();

};
#endif
