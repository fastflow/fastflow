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
#include "utils.h"
#include "PSpecies.h"
#include "SubMatchSet.h"
#include <iostream>
using namespace std;
using namespace cwc_utils;

//constructors
PCompartment::PCompartment (
			    Species *w,
			    Species *cs,
			    vector<PCompartment *> *cc,
			    //variable_symbol &wv,
			    //variable_symbol &cv,
			    variable_adress wv,
			    variable_adress cv,
			    symbol_adress t
			    )
  :
  //wrap(new PSpecies(*w)),
  //content_species(new PSpecies(*cs)),
  content_compartments(cc),
  type(t),
  wrap_variable_symbol(wv),
  content_variable_symbol(cv)
{
  //delete &wv;
  //delete &cv;
  MYNEW(wrap, PSpecies, *w);
  MYNEW(content_species, PSpecies, *cs);
  delete w;
  delete cs;
  sms_cache = NULL;
  if(cc != NULL) {
    //prepare the sms-cache
    n_content_compartments = cc->size();
    if(n_content_compartments > 0) {
      sms_cache = (sms_cache_item_t **)MALLOC(n_content_compartments * sizeof(sms_cache_item_t *));
      for(unsigned int i=0; i<n_content_compartments; ++i)
	sms_cache[i] = NULL;
    }
  }
  else
    n_content_compartments = 0;
  n_subjects_last = 0;
}



//PCompartment::PCompartment () {}



//top level constructor
PCompartment::PCompartment (
			    Species *cs,
			    vector<PCompartment *> *cc,
			    //variable_symbol &cv,
			    variable_adress cv,
			    symbol_adress t
			    )
  :
  wrap(NULL),
  //content_species(new PSpecies(*cs)),
  content_compartments(cc),
  type(t),
  //wrap_variable_symbol(*(wvs_p = new variable_symbol)),
  //MYNEW(wrap_variable_symbol, variable_symbol),
  wrap_variable_symbol(VARIABLE_ADRESS_RESERVED),
  content_variable_symbol(cv)
{
  //delete &cv;
  MYNEW(content_species, PSpecies, *cs);
  delete cs;
  sms_cache = NULL;
  if(cc != NULL) {
    //prepare the sms-cache
    n_content_compartments = cc->size();
    if(n_content_compartments > 0) {
      sms_cache = (sms_cache_item_t **)MALLOC(n_content_compartments * sizeof(sms_cache_item_t *));
      for(unsigned int i=0; i<n_content_compartments; ++i)
	sms_cache[i] = NULL;
    }
  }
  else
    n_content_compartments = 0;
  n_subjects_last = 0;
}



//copy constructor
PCompartment::PCompartment (PCompartment &copy) :
  //content_species(new PSpecies(*copy.content_species)),
  content_compartments(new vector<PCompartment *>),
  type(copy.type),
  //wrap_variable_symbol(*new variable_symbol(copy.wrap_variable_symbol)),
  wrap_variable_symbol(copy.wrap_variable_symbol),
  //content_variable_symbol(*new variable_symbol(copy.content_variable_symbol))
  content_variable_symbol(copy.content_variable_symbol)
{
  MYNEW(content_species, PSpecies, *copy.content_species);
  PSpecies *new_wrap = NULL;
  if(copy.wrap)
    //new_wrap = new PSpecies(*copy.wrap);
    MYNEW(new_wrap, PSpecies, *copy.wrap);
  wrap = new_wrap;
  n_content_compartments = copy.n_content_compartments;
  for(unsigned int i = 0; i < n_content_compartments; ++i) { //content-compartments
    PCompartment *cp;
    //cp = new PCompartment(*copy.content_compartments->at(i));
    MYNEW(cp, PCompartment, *copy.content_compartments->at(i));
    content_compartments->push_back(cp);
  }
  //make an empty sms-cache
  if(n_content_compartments > 0) {
    sms_cache = (sms_cache_item_t **)MALLOC(n_content_compartments * sizeof(sms_cache_item_t *));
    for(unsigned int i=0; i<n_content_compartments; ++i)
      sms_cache[i] = NULL;
  }
  else
    sms_cache = NULL;
  n_subjects_last = 0;
}



//destructor
PCompartment::~PCompartment() {
  //cout << "called on " << *this << endl;
  delete wrap;
  delete content_species;
  if(content_compartments != NULL) {
    for(unsigned int i = 0; i < content_compartments->size(); ++i)
      delete (*content_compartments)[i];
    content_compartments->clear();
    delete content_compartments;
  }
  //delete &wrap_variable_symbol;
  //delete &content_variable_symbol;
  if(sms_cache != NULL)
    FREE(sms_cache);
}

vector<vector<int> > submatchsets(
				  vector<PCompartment *> &objects,
				  unsigned int object_index,
				  vector<Compartment *> &subjects,
				  bool *excluded,
				  sms_cache_item_t **sms_cache
				  )
{
  PCompartment &object(*objects[object_index]);
  vector<vector<int> > res;
  res.reserve(20);

  for(unsigned int subject_index = 0; subject_index < subjects.size(); ++subject_index) {
    if(!excluded[subject_index]) {
      //get (or build) the submatchset
      multiplicityType n = 0;
      if(!sms_cache[object_index][subject_index].sms) {
	SubMatchSet *sms;
	//sms = new SubMatchSet();
	MYNEW(sms, SubMatchSet);
	n = sms->submatch_rule_term(object, *subjects[subject_index], &subjects);
	sms_cache[object_index][subject_index].sms = sms;
      }
      else {
	n = sms_cache[object_index][subject_index].sms->root.multiplicity;
      }
      if(n > 0) {
	//positive submatchset for the object
	unsigned int next_object_index = object_index + 1;
	if(next_object_index < objects.size()) {
	  //next object
	  excluded[subject_index] = true;
	  vector<vector<int> > res_rec = submatchsets(objects, next_object_index, subjects, excluded, sms_cache);
	  excluded[subject_index] = false;
	  //combine results
	  for(vector<vector<int> >::const_iterator it = res_rec.begin(); it != res_rec.end(); ++it) {
	    vector<int> res_item(objects.size() - object_index);
	    res_item[0] = subject_index;
	    unsigned int i = 1;
	    for(vector<int>::const_iterator jt = it->begin(); jt != it->end(); ++jt)
	      res_item[i++] = *jt;
	    res.push_back(res_item);
	  }
	}
	else
	  //last compartment
	  res.push_back(vector<int>(1, subject_index));
      }
    }
  }
  return res;
}

multiplicityType PCompartment::match(Compartment &c, SubMatchSet *sms) {
  if(type != c.type && type != ANY_TYPE) {
    //types don't match
    return 0;
  }

  vector<Compartment *> &subject_cc = *c.content_compartments;
  vector<PCompartment *> &object_cc = *content_compartments;
  unsigned int n_subjects = subject_cc.size();

  if(n_content_compartments > n_subjects)
    //too few subject-compartments
    return 0;

  //wrap
  multiplicityType m_species = 1;
  if(wrap != NULL) {
    if(c.wrap != NULL) {
      if((m_species *= wrap->match(*c.wrap)) == 0)
	//wrap-species don't match
	return 0;
    }
    //not empty object-wrap vs. empty subject-wrap
    else return 0;
  }

  if((m_species *= content_species->match(*c.content_species)) == 0)
    //content-species don't match
    return 0;

  //content-compartments
  multiplicityType m_compartments = 1;
  if(n_content_compartments > 0) {
    //prepare the cache
    for(unsigned int i=0; i<n_content_compartments; ++i) {
      for(unsigned int j=0; j<n_subjects_last; ++j) {
	if(sms_cache[i][j].sms && !sms_cache[i][j].stored)
	  delete sms_cache[i][j].sms;
      }
      if(sms_cache[i])
	FREE(sms_cache[i]);
      //sms_cache[i] = NULL;
      sms_cache[i] = (sms_cache_item_t *)MALLOC(n_subjects * sizeof(sms_cache_item_t));
      for(unsigned int j=0; j<n_subjects; ++j) {
	sms_cache[i][j].sms = NULL;
	sms_cache[i][j].stored = false;
      }
    }
    n_subjects_last = n_subjects;

    //get submatches
    bool *excluded = (bool *)MALLOC(n_subjects * sizeof(bool));
    for(unsigned int i=0; i<n_subjects; ++i)
      excluded[i] = false;
    /*
    int **sms_indexes = (int **)MALLOC(n_content_compartments * sizeof(int *)); //TODO: private field
    for(unsigned int i=0; i<n_content_compartments; ++i)
      sms_indexes[i] = (int *)MALLOC(n_subjects * sizeof(int));
    */
    vector<vector<int> > sms_indexes = submatchsets(object_cc, 0, subject_cc, excluded, sms_cache);
    FREE(excluded);
    excluded = NULL;

    //build results
    m_compartments = 0;
    for(unsigned int i=0; i<sms_indexes.size(); ++i) {
      vector<SubMatchSet **> *content = new vector<SubMatchSet **>(n_content_compartments);
      multiplicityType m_sms = 1;
      for(unsigned int j=0; j<n_content_compartments; ++j) {
	content->at(j) = &(sms_cache[j][sms_indexes[i][j]].sms);
	sms_cache[j][sms_indexes[i][j]].stored = true;
	m_sms *= (*(content->at(j)))->root.multiplicity;
      }
      sms->add_content(content);
      m_compartments += m_sms; 
    }
  }
  else
    //no object-compartments
    m_compartments = 1;

  return m_species * m_compartments;
}



pair<multiplicityType, multiplicityType> PCompartment::match_biochemical(Compartment &c) {
  if(type != c.type)
    return pair<multiplicityType, multiplicityType>(0, 0); //type
  return content_species->match_biochemical(*c.content_species);
}



multiplicityType PCompartment::count(Compartment &c) {
  //type
  if(type != c.type && type != ANY_TYPE)
    return 0;

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
  if(c.wrap) os << *c.wrap;
  os << "~" << c.wrap_variable_symbol;
  os << " | ";
  //content-species
  os << *c.content_species;
  //content-compartments
  for(unsigned int i=0; i<cc.size(); i++) {
    os << *cc[i] << " ";
  }
  os << "~" << c.content_variable_symbol;
  return os << ")";
}
