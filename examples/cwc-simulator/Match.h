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

#ifndef MATCH
#define MATCH
//#include "Compartment.h"
//#include "SubMatchSet.h"
#include "definitions.h"

class SubMatchSet;
class Compartment;

/**
   match of a rule.
   (multiple) match of a rule against a term (on a context), with its stochastic rate
*/
class Match {
 public:
  /**
     the submatchset for the components of the rule
  */
  SubMatchSet &submatchset;
	
  /**
     stochastic rate
  */
  double rate;

  /**
     context
  */
  Compartment *context;

  /**
     the rarest matching multiplicity
  */
  multiplicityType rarest;

  /**
     constructor
  */
  Match(SubMatchSet &submatchset, double rate, Compartment *context, multiplicityType rarest = 0);
  ~Match();

  friend std::ostream& operator<<(std::ostream &, const Match &);
};
#endif
