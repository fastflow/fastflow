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

#include "Model.h"
#include <ostream>
using namespace std;

multiplicityType Model::count(PCompartment &pattern, Compartment &term) {
	
  multiplicityType total = 0;

  //on-level count
  multiplicityType count_level = pattern.count(term);

  //in-depth count
  vector<Compartment *> &object_cc = *term.content_compartments;
  multiplicityType count_rec = 0;
  for (unsigned int i=0; i<object_cc.size(); i++) {
    Compartment &rc = *object_cc[i];
    count_rec += count(pattern, rc);
  }

  //merge
  return total + count_level + count_rec;
}

Model::Model(
	     string &ti,
	     reverse_symbol_table &rst,
	     vector<Rule *> &r,
	     unsigned int ode_size,
	     Species &init_species,
	     vector<Compartment *> &init_compartments,
	     vector<Monitor *> &m
	     )
  : title(ti), reverse_atom_table(rst), rules(r), ode_size(ode_size), term(*new Compartment(&init_species, &init_compartments)), monitors(m) {}

Model::Model(Model &copy) :
  title(*new string(copy.title)),
  reverse_atom_table(*new reverse_symbol_table(copy.reverse_atom_table)),
  rules(*new vector<Rule *>),
  ode_size(copy.ode_size),
  term(*new Compartment(copy.term)),
  monitors(*new vector<Monitor *>)
{
  for(unsigned int i=0; i<copy.rules.size(); i++) rules.push_back(new Rule(*copy.rules[i]));
  for(unsigned int i=0; i<copy.monitors.size(); i++) monitors.push_back(new Monitor(*copy.monitors[i]));
}

Model::~Model() {
  delete &title;
  delete &reverse_atom_table;
  for(unsigned int i = 0; i < rules.size(); ++i)
    delete rules[i];
  delete &rules;
  delete &term;
  for(unsigned int i = 0; i < monitors.size(); ++i)
    delete monitors[i];
  delete &monitors;
}

std::string Model::header(double time_limit) {
  std::stringstream s;
  s.precision(dbl::digits10);
  s << "# Simulation on model '" << title << "' (end at " << std::fixed << time_limit << ")\n";
  s << "# column 1: Time\n";
  for(unsigned int i = 0; i < monitors.size(); ++i)
    s << "# column " << (i + 2) << ": " << monitors[i]->title << "\n";
  s << "\n";
  return s.str();
}

vector<multiplicityType> * Model::monitor() {
  vector<multiplicityType> *res = new vector<multiplicityType>;
  for(unsigned int i=0; i<monitors.size(); i++) {
    res->push_back(count(*monitors[i]->pattern, term));
  }
  return res;
}

std::ostream& operator<<(std::ostream &os, const Model &m) {
  os << "%MODEL: \"" << m.title << "\"\n\n";
  os << "%rules\n";
  for(unsigned int i = 0; i < m.rules.size(); ++i)
    os << *m.rules[i] << "%%\n";
  os << "\n%term\n";
  os << m.term << " ";
  os << "\n\n%monitors\n";
  for(unsigned int i = 0; i < m.monitors.size(); ++i)
    os << *m.monitors[i] << "%%\n";
  return os << "\n%end";
}
