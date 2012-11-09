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

#include "SubMatchSet.h"
//#include "Compartment.h"
#include "PCompartment.h"
//#include "utils.h"
#include <iostream>

SubMatchSet::SubMatchSet() {
  contents.reserve(20); //matches of children compartments
}

SubMatchSet::~SubMatchSet() {
  unsigned int j=0;
  for(contentsType::const_iterator it = contents.begin(); it != contents.end(); ++it, ++j) {
    unsigned int i=0;
    for(vector<SubMatchSet **>::const_iterator jt = (*it)->begin(); jt != (*it)->end(); ++jt, ++i) {
      if(**jt) {
	//cerr << "pre" << endl;
	delete **jt;
	//cerr << "post" << endl;
	*((*it)->at(i)) = NULL;
      }
    }
    contents[j]->clear();
    if(contents[j]) {
      delete contents[j];
      contents[j] = NULL;
    }
  }
  contents.clear();
}

multiplicityType SubMatchSet::submatch(PCompartment &pc, Compartment &tc) {
  return submatch_rule_term(pc, tc, NULL);
}

pair<multiplicityType, multiplicityType> SubMatchSet::submatch_biochemical(PCompartment &pc, Compartment &tc) {
  pair<multiplicityType, multiplicityType> res = pc.match_biochemical(tc);
  if(res.first > 0) {
    root = SubMatch(&pc, &tc, NULL, res.first);
  }
  return res;
}

multiplicityType SubMatchSet::submatch_rule_term(PCompartment &pc, Compartment &tc, vector<Compartment *> *parent) {
  multiplicityType multiplicity = pc.match(tc, this);
  if(multiplicity > 0)
    root = SubMatch(&pc, &tc, parent, multiplicity);
  return multiplicity;
}

void SubMatchSet::add_content(vector<SubMatchSet **> *content) {
  contents.push_back(content);
}

ostream& operator<<(ostream &os, const SubMatchSet &s) {
  os << "<START submatchset" << endl;
  os << "root: " << s.root << endl;
  os << "contents:" << endl;
  for(SubMatchSet::contentsType::const_iterator it = s.contents.begin(); it != s.contents.end(); it++)
    for(vector<SubMatchSet **>::const_iterator jt = (*it)->begin(); jt != (*it)->end(); jt++)
      os << **jt << endl << "-" << endl;
  os << "END submatchset >" << endl;
  return os;
}
