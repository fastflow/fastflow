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

#include "SubMatch.h"

SubMatch::SubMatch(
		   PCompartment *object,
		   Compartment *term,
		   vector<Compartment *> *parent,
		   multiplicityType multiplicity
		   )
  : object(object), term(term), parent(parent), multiplicity(multiplicity) {}

SubMatch::SubMatch() {}



std::ostream& operator<<(ostream &os, const SubMatch &s) {
  os << "\tSM\t" << s.term << ", parent = " << s.parent << ", multiplicity = " << s.multiplicity;
  return os << "\n";
}
