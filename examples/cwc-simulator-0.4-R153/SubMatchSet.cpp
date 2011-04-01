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

SubMatchSet::SubMatchSet() {root = NULL;}

SubMatchSet::~SubMatchSet() {
  delete root;
  for(contentsType::const_iterator it = contents.begin(); it != contents.end(); ++it) {
    vector<SubMatchSet *> &vs = *it->second;
    for(unsigned int i = 0; i < vs.size(); ++i) {
      delete vs[i];
    }
    delete &vs;
  }
}

multiplicityType SubMatchSet::submatch(PCompartment &pc, Compartment &tc) {
  return submatch_rule_term(pc, tc, NULL);
}

pair<multiplicityType, multiplicityType> SubMatchSet::submatch_biochemical(PCompartment &pc, Compartment &tc) {
  pair<multiplicityType, multiplicityType> res = pc.match_biochemical(tc);
  if(res.first > 0) {
    root = new SubMatch(tc, NULL, res.first);
  }
  return res;
}

multiplicityType SubMatchSet::submatch_rule_term(PCompartment &pc, Compartment &tc, vector<Compartment *> *parent) {
  multiplicityType multiplicity = pc.match(tc, this);
  if(multiplicity > 0) {
    root = new SubMatch(tc, parent, multiplicity);
  }
  return multiplicity;
}

void SubMatchSet::add_content(PCompartment *co, SubMatchSet *content) {
  //prepare
  vector<SubMatchSet *> *sv;
  contentsType::const_iterator it = contents.find(co);
  if(it != contents.end()) {
    sv = it->second;
  }
  else {
    sv = new vector<SubMatchSet *>;
    contents[co] = sv;
  }

  //add
  sv->push_back(content);
}

ostream& operator<<(ostream &os, const SubMatchSet &s) {
  os << "root: " << *s.root << endl;
  for(SubMatchSet::contentsType::const_iterator it = s.contents.begin(); it != s.contents.end(); ++it) {
    os << "SMS for compartment \t" << *it->first << ":" << endl;
    for(unsigned int i = 0; i < it->second->size(); ++i)
      os<< *(*it->second)[i];
    os << "\n";
  }
  return os;
}
