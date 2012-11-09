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

#include "Rule.h"
//#include "Compartment.h"
//#include "OCompartment.h"
//#include "PCompartment.h"

Rule::Rule(
	   Species &lhs_s,
	   vector<PCompartment *> &lhs_c,
	   Species &rhs_s,
	   vector<OCompartment *> &rhs_c,
	   semanticsType s,
	   vector<double> p,
	   vector<symbol_adress> *so,
	   //string &lhs_v,
	   //vector<string *> &rhs_v,
	   variable_adress lhs_v,
	   vector<variable_adress> &rhs_v,
	   symbol_adress t
	   )
  :
  lhs(*new PCompartment(&lhs_s, &lhs_c, lhs_v, t)),
  rhs(*new OCompartment(&rhs_s, &rhs_c, rhs_v, t)),
  semantics(s),
  parameters(p),
  type(t),
  is_biochemical(lhs_c.size() == 0 && rhs_c.size() == 0)
{
  if(so) {
    for(unsigned int i=0; i<so->size(); ++i)
      species_occs.push_back(so->at(i));
    delete so;
  }
}

Rule::Rule(Rule &copy) :
  lhs(*new PCompartment(copy.lhs)),
  rhs(*new OCompartment(copy.rhs)),
  semantics(copy.semantics),
  parameters(copy.parameters),
  type(copy.type),
  is_biochemical(copy.is_biochemical),
  ode_index(copy.ode_index),
  species_occs(copy.species_occs) {}

Rule::~Rule() {
  delete &lhs;
  delete &rhs;
}

std::ostream& operator<<(std::ostream &os, const Rule &r) {
  os << r.lhs << " ";
  os << " >>>[" << std::fixed;
  os << " (" << r.semantics << ") ";
  for(unsigned int i=0; i<r.parameters.size(); i++)
    os << r.parameters[i] << " ";
  os << "]>>> ";
  return os << r.rhs;
}
