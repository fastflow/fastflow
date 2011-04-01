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

#ifndef SUBMATCH
#define SUBMATCH
#include "Compartment.h"
#include "definitions.h"
#include <vector>
#include <ostream>
using namespace std;

/**
   (sub)match of simple terms.
   (multiple) match of a simple term against another one
*/
class SubMatch {
 public:
  /**
     matching compartment (subject)
  */
  Compartment &term;

  /**
     parent of the the matching compartment
  */
  vector<Compartment *> *parent;
	
  /**
     multiplicity of the submatch
  */
  multiplicityType multiplicity;

  /**
     constructor
  */
  SubMatch(Compartment &term, vector<Compartment *> *parent, multiplicityType multiplicity);

  friend ostream& operator<<(ostream &, const SubMatch &);
};
#endif
