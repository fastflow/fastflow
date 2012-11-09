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

#ifndef SUBMATCHSET
#define SUBMATCHSET
#include "SubMatch.h"
#include <map>
#include <vector>

class Compartment;
class PCompartment;

#include <ostream>
using namespace std;

class PCompartment;

/**
   matching-tree for a match.
*/
class SubMatchSet {
 public:
  /**
     association between compartments (components of a rule) and their submatches
  */
  typedef vector<vector<SubMatchSet **> *> contentsType;

  SubMatch root;
  contentsType contents;

  /**
     constructor (empty submatchset)
  */
  SubMatchSet();

  ~SubMatchSet();

  /**
     submatching of a compartment against a subject
     @param rule the rule
     @param object the object compartment
     @param subject the subject compartment
     @return the multiplicity of the submatches (-1 if not matching)
  */
  multiplicityType submatch(PCompartment &object, Compartment &subject);

  /**
     submatching of a biochemical compartment against a subject
     @param rule the rule
     @param object the object compartment
     @param subject the subject compartment
     @return the multiplicity of the submatches (-1 if not matching)
  */
  pair<multiplicityType, multiplicityType> submatch_biochemical(PCompartment &object, Compartment &subject);
	
  /**
     basic submatching.
     computes the submatch of a compartment against a subject compartment (from a parent term), updating the submatchset.
     @param rule the rule
     @param object the object compartment
     @param parent the parent (of subject) term
     @param subject the subject compartment
     @return the multiplicity of the submatch (-1 if not matching)
  */
  multiplicityType submatch_rule_term(PCompartment &object, Compartment &subject, vector<Compartment *> *parent);

  void add_content(vector<SubMatchSet **> *);

  friend ostream& operator<<(ostream &, const SubMatchSet &);
};
#endif
