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

#include "Gillespie.h"
#include "definitions.h"
#include "Compartment.h"
#include "Instantiation.h"
#include "Match.h"
#include "OCompartment.h"
#include "PCompartment.h"
#include "Rule.h"
#include "SubMatch.h"
#include "SubMatchSet.h"
#ifdef HYBRID
#include "ode.h"
#endif
#include <map>
#include <typeinfo>
//#include <iostream>

using namespace std;

typedef map<PCompartment *, SubMatch *> updatesType;

//utility: find a compartment in a vector of terms
int find_in_ptr_vector(vector<Compartment *> &v, Compartment *p) {
  for(unsigned int i = 0; i < v.size(); ++i)
    if(v[i] == p) return i;
  return -1;
}

//stochastic walk on the matchset (and submatchset)
pair<Rule*, Match *> get_mu(MatchSet &m, double d) {
  double s = 0;
  for(MatchSet::matchsetType::const_iterator it = m.matchset.begin(); it != m.matchset.end(); ++it) {
    for(unsigned int i = 0; i < it->second->size(); ++i) { //matches
      s += (*it->second)[i]->rate;
      if(d < s) {
	//picked rule-and-match
	return make_pair(it->first, (*it->second)[i]);
      }
    }
  }
  return make_pair((Rule *)NULL, (Match *)NULL);
}

SubMatchSet* resolve_contents(vector<SubMatchSet *> &contents, u01_vg_type &rng) {
  unsigned int nc = contents.size();  
  if(nc == 1) {
    return contents[0];
  }
  else {
    //TODO: track in submatchset
    multiplicityType sum = 0;
    for(unsigned int i = 0; i < nc; ++i) {
      sum += contents[i]->root->multiplicity;
    }
    double d = rng() * sum;
    unsigned long long s = 0;
    unsigned int i = 0;
    for(; i < nc; ++i) {
      s += contents[i]->root->multiplicity;
      if(d < double(s)) {
	break;
      }
    }
    return contents[i];
  }
}

void get_updates_(updatesType &updates, SubMatchSet &s, u01_vg_type &rng, PCompartment *root) {
  //root update
  updates[root] = s.root;
  //other updates
  for(SubMatchSet::contentsType::const_iterator it = s.contents.begin(); it != s.contents.end(); ++it) {
    PCompartment *pc = it->first;
    SubMatchSet &sms = *resolve_contents(*it->second, rng);
    get_updates_(updates, sms, rng, pc);
  }
}

updatesType* get_updates(SubMatchSet &s, u01_vg_type &rng, PCompartment *rule_lhs_root) {
  updatesType &updates = *new updatesType;
  get_updates_(updates, s, rng, rule_lhs_root);
  return &updates;
}

#ifdef HYBRID
void apply_ode(MatchSet::odeRatesType &ode, double tau) {
  for(MatchSet::odeRatesType::const_iterator it = ode.begin(); it != ode.end(); ++it) {
    Compartment &c = *it->first;
    vector<double> &k = *it->second;
    limited_ode(k, *c.content_species, tau);
  }
}

void retry(MatchSet &ma, double &time, u01_vg_type &rng) {
  //update the time
  double tau = (1 / ma.rate_sum_deterministic) * log(1 / rng());
  time += tau;

  //ODE
  apply_ode(ma.ode_rates, tau);
}
#endif

void do_updates(Rule &r, updatesType &updates,
#ifdef HYBRID
		MatchSet::odeRatesType &ode,
		double tau,
#endif
		Compartment &context) {
  vector<Compartment *> compartments_to_delete;
  instantiationType instantiations;

  //RESERVE
  PCompartment &rule_lhs_root(r.lhs);
  SubMatch &top_submatch(*updates[&rule_lhs_root]); //top level submatch
  Compartment &c(top_submatch.term);
  c.update_delete((Compartment &)rule_lhs_root); //delete at top level
  //cout << "trunked term: " << *c << endl;
  instantiations[rule_lhs_root.content_variable_symbol] = make_pair(c.content_species, c.content_compartments); //reference top level X
  updates.erase(&rule_lhs_root); //remove the visited submatch
	
  //nested compartments
  for(updatesType::const_iterator it = updates.begin(); it != updates.end(); ++it) {
    //get a submatch and delete reactants
    PCompartment &pc(*it->first);
    SubMatch &s(*it->second);
    Compartment &c(s.term);
    c.update_delete((Compartment &)pc);
    //reference variables
    instantiations[pc.content_variable_symbol] = make_pair(c.content_species, c.content_compartments);
    instantiations[pc.wrap_variable_symbol] = make_pair(c.wrap, (vector<Compartment *> *)NULL);
    //remove the compartment
    int position_to_delete = find_in_ptr_vector(*s.parent, &s.term);
    compartments_to_delete.push_back(&c);
    s.parent->erase(s.parent->begin() + position_to_delete);
  }

#ifdef HYBRID
  //ODE
  apply_ode(ode, tau);
#endif
  
  //ADD
  Compartment &fresh_c = *r.rhs.ground(instantiations);
  context.replace_content(fresh_c.content_species, fresh_c.content_compartments);

  // clean-up
  for(unsigned int i = 0; i < compartments_to_delete.size(); ++i)
    delete compartments_to_delete[i];
  fresh_c.content_species = NULL;
  fresh_c.content_compartments = NULL;
  delete &fresh_c;

}

#ifdef HYBRID
void hybrid_gillespie(MatchSet &ma, double &time, u01_vg_type &rng) {
  //update the time
  double tau = (1 / ma.rate_sum) * log(1 / rng());
  time += tau;

  //decide a match and get the context
  pair<Rule *, Match *> mu = get_mu(ma, rng() * ma.rate_sum);
  Compartment *context = mu.second->context;
  Rule &rule = *mu.first;
  //cerr << "picked rule: " << *mu.first << endl;

  //compute detailed updates and apply
  updatesType *updates = get_updates(mu.second->submatchset, rng, &rule.lhs);
  do_updates(rule, *updates, ma.ode_rates, tau, *context);

  //clean-up
  delete updates;
}

#else
void stochastic_gillespie(MatchSet &ma, double &time, u01_vg_type &rng) {
  //update the time
  double tau = (1 / ma.rate_sum) * log(1 / rng());
  time += tau;

  //decide a match and get the context
  pair<Rule *, Match *> mu = get_mu(ma, rng() * ma.rate_sum);
  Compartment *context = mu.second->context;
  Rule &rule = *mu.first;
  //cout << "picked rule: " << *mu.first << endl;

  //compute detailed updates and apply
  updatesType *updates = get_updates(mu.second->submatchset, rng, &rule.lhs);
  do_updates(rule, *updates, *context);

  //clean-up
  delete updates;
}
#endif
