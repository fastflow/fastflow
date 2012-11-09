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
#include "utils.h"
//#include "PCompartment.h"
//#include "OCompartment.h"
//#include <typeinfo>


//constructor
Compartment::Compartment(Species *w, Species *cs, vector<Compartment *> *cc, symbol_adress t)
  : wrap(w), content_species(cs), content_compartments(cc), type(t) {}



//top level constructor
Compartment::Compartment(Species *cs, vector<Compartment *> *cc, symbol_adress t)
  : wrap(NULL), content_species(cs), content_compartments(cc), type(t) {}



//copy constructor
Compartment::Compartment(Compartment &copy) :
  //content_species(new Species(*copy.content_species)),
  content_compartments(new vector<Compartment *>),
  type(copy.type)
{
  MYNEW(content_species, Species, *copy.content_species);
  Species *new_wrap = NULL;
  if (copy.wrap)
    //new_wrap = new Species(*copy.wrap);
    MYNEW(new_wrap, Species, *copy.wrap);
  wrap = new_wrap;
  for(unsigned int i = 0; i < copy.content_compartments->size(); ++i) { //content-compartments
    Compartment *cp;
    //cp = new Compartment(*copy.content_compartments->at(i));
    MYNEW(cp, Compartment , *copy.content_compartments->at(i));
    content_compartments->push_back(cp);
  }
}


void Compartment::delete_content() {
  delete content_species;
  content_species = NULL;
  if(content_compartments != NULL) {
    for(unsigned int i = 0; i < content_compartments->size(); ++i) {
      delete content_compartments->at(i);
    }
    content_compartments->clear();
    delete content_compartments;
    content_compartments = NULL;
  }
}

//destructor
Compartment::~Compartment() {
  delete wrap;
  wrap = NULL;
  delete_content();
}



void Compartment::update_delete(Compartment &c) {
  //wrap
  if(c.wrap != NULL && wrap != NULL)
    wrap->update_delete(*c.wrap);

  //content-species
  content_species->update_delete(*c.content_species);
}



void Compartment::replace_content(Species *ns, vector<Compartment *> *nc) {
  delete_content();
  content_species = ns;
  content_compartments = nc;
}



void Compartment::add_content_species(Species &delta) {
  content_species->update_add(delta);
}

void Compartment::trunked_content(Compartment &object, Species &diff) {
  content_species->diff(*(object.content_species), diff);
}



ostream& operator<<(ostream &os, Compartment &c) {
  os << "( {" << &c << "}";
  //type
  os << "[ " << c.type << " ] ";
  //wrap
  if(c.wrap)
    os << *(c.wrap);
  os << " | ";
  //content-species
  if(c.content_species)
    os << *c.content_species;
  //content-compartments
  if(c.content_compartments) {
    vector<Compartment *> &cc = *c.content_compartments;
    for(unsigned int i=0; i<cc.size(); i++) {
      os << *cc[i] << " ";
    }
  }
  return os << ")";
}
