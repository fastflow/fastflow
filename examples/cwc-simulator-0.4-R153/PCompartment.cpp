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

#include "PCompartment.h"
#include <iostream>
using namespace std;



//constructor
PCompartment::PCompartment (
			    Species *w,
			    Species *cs,
			    vector<PCompartment *> *cc,
			    variable_symbol &wv,
			    variable_symbol &cv,
			    symbol_adress t
			    )
  : wrap(new PSpecies(*w)), content_species(new PSpecies(*cs)), content_compartments(cc), type(t), wrap_variable_symbol(wv), content_variable_symbol(cv)
{
  delete w;
  delete cs;
}



//top level constructor
PCompartment::PCompartment (
			    Species *cs,
			    vector<PCompartment *> *cc,
			    variable_symbol &cv,
			    symbol_adress t
			    )
  : wrap(NULL), content_species(new PSpecies(*cs)), content_compartments(cc), type(t), wrap_variable_symbol(*new variable_symbol), content_variable_symbol(cv)
{
  delete cs;
}



//copy constructor
PCompartment::PCompartment (PCompartment &copy) :
  content_species(new PSpecies(*copy.content_species)),
  content_compartments(new vector<PCompartment *>),
  type(copy.type),
  wrap_variable_symbol(*new variable_symbol(copy.wrap_variable_symbol)),
  content_variable_symbol(*new variable_symbol(copy.content_variable_symbol))
{
  PSpecies *new_wrap = (copy.wrap != NULL)? new PSpecies(*copy.wrap) : NULL;
  wrap = new_wrap;
  for(unsigned int i = 0; i < copy.content_compartments->size(); ++i) //content-compartments
    content_compartments->push_back(new PCompartment(*copy.content_compartments->at(i)));
}



//destructor
PCompartment::~PCompartment() {
  //cout << "called on " << *this << endl;
  delete wrap;
  delete content_species;
  if(content_compartments != NULL) {
    for(unsigned int i = 0; i < content_compartments->size(); ++i) {
      delete (*content_compartments)[i];
    }
    delete content_compartments;
  }
  delete &wrap_variable_symbol;
  delete &content_variable_symbol;
}


multiplicityType PCompartment::match(Compartment &c, SubMatchSet *sms) {
  //type
  if(type != c.type) return 0;

  vector<Compartment *> &subject_cc = *c.content_compartments;
  vector<PCompartment *> &object_cc = *content_compartments;

  //content-compartments size
  if(object_cc.size() > subject_cc.size()) return 0;

  //wrap
  multiplicityType n = 1;
  if(wrap != NULL) {
    if(c.wrap != NULL) {
      if((n *= wrap->match(*c.wrap)) == 0) return 0;
    }
    else return 0;
  }

  //content-species
  if((n *= content_species->match(*c.content_species)) == 0) return 0;

  //content-compartments
  multiplicityType cc_n = 1;
  for(unsigned int i = 0; i < object_cc.size(); i++) { //object compartments
    PCompartment &object = *object_cc[i];
    multiplicityType c_n = 0;

    for(unsigned int k = 0; k < subject_cc.size(); ++k) { //subject compartments
      Compartment &subject = *subject_cc[k];
      SubMatchSet *sms_content_ik = new SubMatchSet;
      multiplicityType r_ik = sms_content_ik->submatch_rule_term(object, subject, c.content_compartments);
      if(r_ik > 0) {
	//found match
	sms->add_content(&object, sms_content_ik);
	c_n += r_ik;
      }
      else {
	//delete the empty submatchset
	delete sms_content_ik;
      }
    }

    //works only if there is at most one object-compartment
    cc_n = c_n;
  }

  return n * cc_n;
}



pair<multiplicityType, multiplicityType> PCompartment::match_biochemical(Compartment &c) {
  if(type != c.type) return pair<multiplicityType, multiplicityType>(0, 0); //type
  return content_species->match_biochemical(*c.content_species);
}



multiplicityType PCompartment::count(Compartment &c) {
  //type
  if(type != c.type) return 0;

  vector<Compartment *> &subject_cc = *c.content_compartments;
  vector<PCompartment *> &object_cc = *content_compartments;

  //content-compartments size
  if(object_cc.size() > subject_cc.size()) return 0;

  //wrap
  multiplicityType n = 1;
  if(wrap != NULL) {
    if(c.wrap != NULL) {
      if((n *= wrap->match(*c.wrap)) == 0) return 0;
    }
    else return 0;
  }

  //content-species
  if((n *= content_species->match(*c.content_species)) == 0) return 0;

  //content-compartments
  multiplicityType cc_n = 1;
  for(unsigned int i = 0; i < object_cc.size(); i++) { //object compartments
    PCompartment &object = *object_cc[i];
    multiplicityType c_n = 0;

    for(unsigned int k = 0; k < subject_cc.size(); ++k) { //subject compartments
      Compartment &subject = *subject_cc[k];
      c_n += object.count(subject);
    }

    //works only if there is at most one object-compartment
    cc_n = c_n;
  }

  return n * cc_n;
}



ostream& operator<<(ostream &os, PCompartment &c) {
  vector<PCompartment *> &cc = *c.content_compartments;
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
