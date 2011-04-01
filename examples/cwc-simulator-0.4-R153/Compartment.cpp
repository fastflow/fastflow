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

#include "Compartment.h"
#include "PCompartment.h"
#include "OCompartment.h"
#include <typeinfo>


//constructor
Compartment::Compartment(Species *w, Species *cs, vector<Compartment *> *cc, symbol_adress t)
  : wrap(w), content_species(cs), content_compartments(cc), type(t) {}



//top level constructor
Compartment::Compartment(Species *cs, vector<Compartment *> *cc, symbol_adress t)
  : wrap(NULL), content_species(cs), content_compartments(cc), type(t) {}



//copy constructor
Compartment::Compartment(Compartment &copy) :
  content_species(new Species(*copy.content_species)),
  content_compartments(new vector<Compartment *>),
  type(copy.type)
{
  Species *new_wrap = (copy.wrap != NULL)? new Species(*copy.wrap) : NULL;
  wrap = new_wrap;
  for(unsigned int i = 0; i < copy.content_compartments->size(); ++i) { //content-compartments
    content_compartments->push_back(new Compartment(*copy.content_compartments->at(i)));
  }
}


void Compartment::delete_content() {
  delete content_species;
  if(content_compartments != NULL) {
    for(unsigned int i = 0; i < content_compartments->size(); ++i) {
      delete (*content_compartments)[i];
    }
    delete content_compartments;
  }
}

//destructor
Compartment::~Compartment() {
  delete wrap;
  delete_content();
}



void Compartment::update_delete(Compartment &c) {
  //wrap
  if(wrap != NULL && c.wrap != NULL) wrap->update_delete(*c.wrap);

  //content-species
  content_species->update_delete(*c.content_species);
}



void Compartment::replace_content(Species *ns, vector<Compartment *> *nc) {
  delete_content();
  content_species = ns;
  content_compartments = nc;
}



ostream& operator<<(ostream &os, Compartment &c) {
  vector<Compartment *> &cc = *c.content_compartments;
  os << "( ";
  //type
  os << "[ " << c.type << " ] ";
  //wrap
  if(c.wrap != NULL) os << *c.wrap;
  os << " | ";
  //content-species
  os << *c.content_species;
  //content-compartments
  for(unsigned int i=0; i<cc.size(); i++) {
    os << *cc[i] << " ";
  }
  return os << ")";
}
