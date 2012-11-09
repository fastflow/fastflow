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

#include "OCompartment.h"
#include "utils.h"
//#include <iostream>
//using namespace std;

//constructor
OCompartment::OCompartment(
			   Species *w,
			   Species *cs,
			   vector<OCompartment *> *cc,
			   //vector<variable_symbol *> &wv,
			   //vector<variable_symbol *> &cv,
			   vector<variable_adress> wv,
			   vector<variable_adress> cv,
			   symbol_adress t
			   )
  :  wrap(w), content_species(cs), content_compartments(cc), type(t), wrap_variables_symbols(wv), content_variables_symbols(cv) {}

//top level constructor
OCompartment::OCompartment(
			   Species *cs,
			   vector<OCompartment *> *cc,
			   //vector<variable_symbol *> &cv,
			   vector<variable_adress> cv,
			   symbol_adress t
			   )
  : wrap(NULL), content_species(cs), content_compartments(cc), type(t),
    //wrap_variables_symbols(*new vector<variable_symbol *>),
    //wrap_variables_symbols(*new vector<variable_adress>),
    content_variables_symbols(cv)
{
  wrap_variables_symbols.push_back(VARIABLE_ADRESS_RESERVED);
}

//copy constructor
OCompartment::OCompartment (OCompartment &copy) :
  //content_species(new Species(*copy.content_species)),
  content_compartments(new vector<OCompartment *>),
  type(copy.type)
  //wrap_variables_symbols(*new vector<variable_symbol *>),
  //content_variables_symbols(*new vector<variable_symbol *>)
  //wrap_variables_symbols(*new vector<variable_adress>),
  //content_variables_symbols(*new vector<variable_adress>)
{
  MYNEW(content_species, Species, *copy.content_species);
  //Species *new_wrap = (copy.wrap != NULL)? new Species(*copy.wrap) : NULL;
  Species *new_wrap = NULL;
  if(copy.wrap)
    MYNEW(new_wrap, Species, *copy.wrap);
  wrap = new_wrap;
  for(unsigned int i = 0; i < copy.content_compartments->size(); ++i) { //content-compartments
    OCompartment *ocp;
    MYNEW(ocp, OCompartment, *copy.content_compartments->at(i));
    content_compartments->push_back(ocp);
  }
  for(unsigned int i = 0; i < copy.wrap_variables_symbols.size(); ++i) { //wrap-variables
    /*
    variable_symbol *vsp;
    MYNEW(vsp, variable_symbol, *copy.wrap_variables_symbols[i]);
    wrap_variables_symbols.push_back(vsp);
    */
    wrap_variables_symbols.push_back(copy.wrap_variables_symbols[i]);
  }
  for(unsigned int i = 0; i < copy.content_variables_symbols.size(); ++i) { //content-variables
    /*
    variable_symbol *vsp;
    MYNEW(vsp, variable_symbol, *copy.content_variables_symbols[i]);
    content_variables_symbols.push_back(vsp);
    */
    content_variables_symbols.push_back(copy.content_variables_symbols[i]);
  }
  /*
  for(unsigned int i = 0; i < copy.content_compartments->size(); ++i) //content-compartments
    content_compartments->push_back(new OCompartment(*copy.content_compartments->at(i)));
  for(unsigned int i = 0; i < copy.wrap_variables_symbols.size(); ++i) //wrap-variables
    wrap_variables_symbols.push_back(new variable_symbol(*copy.wrap_variables_symbols[i]));
  for(unsigned int i = 0; i < copy.content_variables_symbols.size(); ++i) //content-variables
    content_variables_symbols.push_back(new variable_symbol(*copy.content_variables_symbols[i]));
  */
}


//destructor
OCompartment::~OCompartment() {
  delete wrap;
  delete content_species;
  if(content_compartments != NULL) {
    for(unsigned int i = 0; i < content_compartments->size(); ++i) {
      delete (*content_compartments)[i];
    }
    content_compartments->clear();
    delete content_compartments;
  }
  //wrap variables
  /*
  for(unsigned int i = 0; i < wrap_variables_symbols.size(); ++i) {
    delete wrap_variables_symbols[i];
  }
  wrap_variables_symbols.clear();
  */
  //delete &wrap_variables_symbols;
  //content variables
  /*
  for(unsigned int i = 0; i < content_variables_symbols.size(); ++i) {
    delete content_variables_symbols[i];
  }
  content_variables_symbols.clear();
  */
  //delete &content_variables_symbols;
}

Compartment * OCompartment::ground(instantiationType &i) {
  //wrap
  Species *new_wrap = NULL;
  if(wrap != NULL) {
    //new_wrap = new Species(*wrap); //explicit
    MYNEW(new_wrap, Species, *wrap);
    for(unsigned int j = 0; j < wrap_variables_symbols.size(); ++j) { //variables
      instantiationType::const_iterator it = i.find(wrap_variables_symbols[j]); //find the instantiation
      Species *instatiated = (it->second).first; //get the species
      new_wrap->update_add(*instatiated); //add the instantiated wrap variable
    }
  }

  vector<OCompartment *> &explicit_cc = *content_compartments;

  //compute the number of content-compartments
  unsigned int new_cc_i = 0;
  unsigned int new_cc_size = explicit_cc.size();
  for(unsigned int j=0; j<content_variables_symbols.size(); j++) {
    instantiationType::const_iterator it = i.find(content_variables_symbols[j]); //find the instantiation
    vector<Compartment *> &instantiated_compartments = *((it->second).second); //get the compartments
    new_cc_size += instantiated_compartments.size();
  }

  //explicit content
  Species *nsp;
  //nsp = new Species(*content_species);
  MYNEW(nsp, Species, *content_species);
  Species &new_cs = *nsp;
  vector<Compartment *> &new_cc = *new vector<Compartment *>(new_cc_size);
  for(unsigned int j=0; j<explicit_cc.size(); ++j) {
    //new_cc.push_back((*explicit_cc[j]).ground(i));
    new_cc[new_cc_i++] = (*explicit_cc[j]).ground(i);
  }

  //content variables
  for(unsigned int j=0; j<content_variables_symbols.size(); j++) {
    instantiationType::const_iterator it = i.find(content_variables_symbols[j]); //find the instantiation
    Species &instantiated_species = *((it->second).first); //get the species
    //cout << "instantiating " << *content_variables_symbols[j] << " to: " << instantiated_species << " ";
    new_cs.update_add(instantiated_species);
    vector<Compartment *> &instantiated_compartments = *((it->second).second); //get the compartments
    for(unsigned int k=0; k<instantiated_compartments.size(); k++) {
      Compartment *nc;
      //nc = new Compartment(*instantiated_compartments[k]);
      MYNEW(nc, Compartment, *instantiated_compartments[k]);
      //cout << *nc << " ";
      //new_cc.push_back(nc);
      new_cc[new_cc_i++] = nc;
    }
    //std::cout << "Size " << new_cc.size() << "\n";
    //cout << endl;
  }

  Compartment *outp;
  //outp = new Compartment(new_wrap, &new_cs, &new_cc, type);
  MYNEW(outp, Compartment, new_wrap, &new_cs, &new_cc, type);
  return outp;
}

ostream& operator<<(ostream &os, OCompartment &c) {
  vector<OCompartment *> &cc = *c.content_compartments;
  os << "( ";
  //type
  os << "[ " << c.type << " ] ";
  //wrap
  if(c.wrap != NULL) os << *c.wrap;
  for(unsigned int i=0; i<c.wrap_variables_symbols.size(); ++i)
    os << "~" << c.wrap_variables_symbols[i] << " ";
  os << " | ";
  //content-species
  os << *c.content_species;
  //content-compartments
  for(unsigned int i=0; i<cc.size(); i++) {
    os << *cc[i] << " ";
  }
  for(unsigned int i=0; i<c.content_variables_symbols.size(); ++i)
    os << "~" << c.content_variables_symbols[i] << " ";
  return os << ")";
}
