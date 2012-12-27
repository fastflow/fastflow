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

#ifndef _CWC_GILLESPIE_HPP_
#define _CWC_GILLESPIE_HPP_
#include "random.h"
#include "MatchSet.h"
//#include "definitions.h"
//#include "Compartment.h"
//#include "Instantiation.h"
#include "Match.h"
//#include "OCompartment.h"
//#include "PCompartment.h"
#include "Rule.h"
//#include "SubMatch.h"
#include "SubMatchSet.h"
#include "utils.h"
#ifdef HYBRID
#include "ode.h"
#endif
#include <map>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
using namespace std;

//stochastic updates
typedef map<PCompartment *, SubMatch *> updatesType;
typedef struct updates {
  Rule *rule;
  Compartment *context;
  updatesType *s_updates;
} updates_t;

//ode updates
typedef map<Compartment *, vector<Species *> > ode_table_t;
typedef struct ode_delta {
  double t_start, t_end, t_step, t_sampling;
  unsigned int n_fired_steps;
  ode_table_t table;

  ode_delta() {
    n_fired_steps = 0;
  }

  ~ode_delta() {
    clear();
  }

  void clear() {
    for(ode_table_t::iterator it = table.begin(); it != table.end(); ++it)
      for(unsigned int i=0; i<it->second.size(); ++i)
	delete it->second[i];
    table.clear();
    n_fired_steps = 0;
  }
} ode_delta_t;

class Gillespie {
public:
  Gillespie(int seed
#ifdef LOG
	    , ofstream &logfile
#endif
	    )
#ifdef LOG
    : logfile(logfile)
#endif
  {
    //rng = new u01_vg_type(rng_a_type(seed), u01_gm_type());
    MYNEW(rng, u01_vg_type, rng_a_type(seed), u01_gm_type());
  }

  ~Gillespie() {
    delete rng;
  }

  double tau(MatchSet &ma) {
    return (1 / ma.rate_sum) * log(1 / (*rng)());
  }

  double tau_retry(MatchSet &ma) {
    return (1 / ma.rate_sum_deterministic) * log(1 / (*rng)());
  }

  void compute_updates(updates_t &out, MatchSet &ma) {
#ifdef HYBRID
    if(!ma.ode_only) {
#endif
      
      //decide a match and get the context
      pair<Rule *, Match *> mu = get_mu(ma, (*rng)() * ma.rate_sum);
      
      /*
	#ifdef LOG
	logfile << "picked rule: " << *mu.first << endl;
	#endif
      */
      
      //compute detailed updates
      Rule *rule = mu.first;
      out.s_updates = get_updates(mu.second->submatchset, &(rule->lhs));
      out.rule = rule;
      out.context = mu.second->context;

#ifdef HYBRID
    }
#endif

  }

#ifdef HYBRID
  void compute_ode_delta(ode_delta_t &out, MatchSet &ma, updates_t &u, double tau, unsigned int n_species) {
    /*
      unsigned int n_ode_steps = NPoint - 1;
      out.t_step = tau / n_ode_steps;
    */
    out.t_step = min(tau, out.t_sampling);
    // unsigned int n_ode_steps = (unsigned int)(tau / out.t_step);
    // unsigned int n_ode_points = n_ode_steps + 1;
    unsigned int n_ode_points = (unsigned int)floor(tau / out.t_step) + 1 + (unsigned int)(fmod(tau, out.t_step)>0.0);
#ifdef LOG
    logfile << "< ODE-delta" << endl;
    logfile << "tau: " << tau
	    << ", start: " << out.t_start
	    << ", end: " << out.t_end
	    << ", step: " << out.t_step << endl;
#endif //LOG

    //allocate the table
    for(MatchSet::odeRatesType::iterator it = ma.ode_rates.begin(); it != ma.ode_rates.end(); ++it) {
      out.table[it->first] = vector<Species *>(n_ode_points);
      for(unsigned int i=0; i<n_ode_points; ++i) {
	//out.table[it->first][i] = new Species(n_species);
	MYNEW(out.table[it->first][i], Species, n_species);
      }
    }

    //allocate the concentrations for the solver
    data = (double **)malloc((n_ode_points) * sizeof(double *));
    for(unsigned int i=0; i<(n_ode_points); i++)
      data[i] = (double *)malloc(n_species * sizeof(double));



    //ODE-delta for each compartment
    if(!ma.ode_only) {
      //WITH stochastic reservation
      updatesType &updates(*(u.updates));
      for(updatesType::const_iterator it = updates.begin(); it != updates.end(); ++it) {
	//get a submatch
	PCompartment &pc(*it->first);
	SubMatch &s(*it->second);
	Compartment &c(*s.term);
	if(ma.ode_rates.count(&c)) {
	  //compute the diff (stochastic reservation)
	  c.trunked_content((Compartment &)pc, *(out.table[&c][0]));

	  //cast rates to array
	  vector<double> &k(*(ma.ode_rates[&c]));
	  unsigned int nk(k.size());
	  double *k_array = (double *)malloc(nk * sizeof(double));
	  for(unsigned int i=0; i<nk; i++)
	    k_array[i] = k[i];
	
	  //get concentrations (from multiplicities)	
	  for(unsigned int i=0; i<n_species; i++)
	    data[0][i] = double((*(out.table[&c][0]))[i]);
	
	  //ODE
#ifdef LOG
	  logfile << "$ {" << &c << "} calling solver with:" << endl
		  << "data:" << endl;
	  for(unsigned int i=0; i<n_ode_points; i++) {
	    logfile << "[ ";
	    for(unsigned int j=0; j<n_species; j++)
	      logfile << data[i][j] << " ";
	    logfile << "]" << endl;
	  }
	  logfile	<< "k_array: [ ";
	  for(unsigned int i=0; i<nk; i++)
	    logfile << k_array[i] << " ";
	  logfile << "]" << endl;
	  logfile	<< "t0: " << 0.0 << endl
			<< "tf: " << tau << endl
			<< "delta: " << out.t_step << endl
			<< "n_species: " << n_species << endl
			<< "n_ode_points: " << n_ode_points << endl;
#endif //LOG
	  odeSolver(data, k_array, 0.0, tau, out.t_step, n_species);
#ifdef LOG
	  logfile << "$ {" << &c << "} solver output:" << endl;
	
	  for(unsigned int i=0; i<n_ode_points; i++) {
	    logfile << "[ ";
	    for(unsigned int j=0; j<n_species; j++)
	      logfile << data[i][j] << " ";
	    logfile << "]" << endl;
	  }
#endif //LOG
	
	  //set multiplicities (from concentrations)
	  for(unsigned int i=1; i<n_ode_points; ++i)
	    for(unsigned int j=0; j<n_species; ++j)
	      (*(out.table[&c][i]))[j] = my_round(data[i][j]);
	
	  //compute delta from starting point
	  for(unsigned int i=n_ode_points-1; i>0; --i)
	    for(unsigned int j=0; j<n_species; ++j) {
	      (*(out.table[&c][i]))[j] -= (*(out.table[&c][i-1]))[j];
	    }
	  //set 0-delta for the starting point
	  for(unsigned int j=0; j<n_species; ++j) {
	    (*(out.table[&c][0]))[j] = 0;
	  }

	  //clean
	  free(k_array);
	}
      }
    }

    else {
      //WITHOUT stochastic reservation
      //updatesType &updates(*(u.updates));
      for(MatchSet::odeRatesType::iterator it = ma.ode_rates.begin(); it != ma.ode_rates.end(); ++it) {
	//get the subject compartment
	Compartment &c(*(it->first));
	Species &s(*(c.content_species));
	for(unsigned int i=0; i<n_species; ++i)
	  (*(out.table[&c][0]))[i] = s[i];
	
	//cast rates to array
	vector<double> &k(*(it->second));
	unsigned int nk(k.size());
	double *k_array = (double *)malloc(nk * sizeof(double));
	for(unsigned int i=0; i<nk; i++)
	  k_array[i] = k[i];
	
	//get concentrations (from multiplicities)	
	for(unsigned int i=0; i<n_species; i++)
	  data[0][i] = (double)((*(out.table[&c][0]))[i]);
	
	//ODE
#ifdef LOG
	logfile << "$ {" << &c << "} calling solver with:" << endl
		<< "data:" << endl;
	for(unsigned int i=0; i<n_ode_points; i++) {
	  logfile << "[ ";
	  for(unsigned int j=0; j<n_species; j++)
	    logfile << data[i][j] << " ";
	  logfile << "]" << endl;
	}
	logfile	<< "k_array: [ ";
	for(unsigned int i=0; i<nk; i++)
	  logfile << k_array[i] << " ";
	logfile << "]" << endl;
	logfile	<< "t0: " << 0.0 << endl
		<< "tf: " << tau << endl
		<< "delta: " << out.t_step << endl
		<< "n_species: " << n_species << endl
		<< "n_ode_points: " << n_ode_points << endl;
#endif //LOG
	odeSolver(data, k_array, 0.0, tau, out.t_step, n_species);
#ifdef LOG
	logfile << "$ {" << &c << "} solver output:" << endl;
	
	for(unsigned int i=0; i<n_ode_points; i++) {
	  logfile << "[ ";
	  for(unsigned int j=0; j<n_species; j++)
	    logfile << data[i][j] << " ";
	  logfile << "]" << endl;
	}
#endif //LOG
	
	//set multiplicities (from concentrations)
	for(unsigned int i=1; i<n_ode_points; ++i)
	  for(unsigned int j=0; j<n_species; ++j)
	    (*(out.table[&c][i]))[j] = my_round(data[i][j]);
	
	//compute delta from starting point
	for(unsigned int i=n_ode_points-1; i>0; --i)
	  for(unsigned int j=0; j<n_species; ++j) {
	    (*(out.table[&c][i]))[j] -= (*(out.table[&c][i-1]))[j];
	  }
	//set 0-delta for the starting point
	for(unsigned int j=0; j<n_species; ++j) {
	  (*(out.table[&c][0]))[j] = 0;
	}
	
	//clean
	free(k_array);
      }
    }
  
    //clean
    for(unsigned int i=0; i<n_ode_points/* + 10*/; i++)
      free(data[i]);
    free(data);
    
#ifdef LOG
    for(ode_table_t::iterator it = out.table.begin(); it != out.table.end(); ++it) {
      logfile << *(it->first) << ": ";
      for(unsigned int i=0; i<n_ode_points; ++i)
	logfile << *(it->second[i]) << fixed << "| ";
      logfile << endl;
    }
    logfile << "ODE-delta >" << endl;
#endif //LOG
  }
#endif //HYBRID

  void update(MatchSet &ma, updates_t &u
#ifdef HYBRID
	      , ode_delta_t &ode_delta
#endif
	      ) {
#ifdef HYBRID
    if(!ma.ode_only) {
#endif
      
      do_updates(*(u.rule), *(u.s_updates)
#ifdef HYBRID
		 , ode_delta
#endif
		 , *(u.context));
      
      //clean-up
      delete u.s_updates;
      
#ifdef HYBRID
    }
    //ode only
    else
      ode(ode_delta);
#endif
  }

#ifdef HYBRID
  void ode(ode_delta_t &ode_delta, double t_target) {
#ifdef LOG
    logfile << "apply ODE to " << t_target << ":";
#endif
    double t_next_step;
    while(true) {
      t_next_step = ode_delta.t_start + ode_delta.n_fired_steps * ode_delta.t_step;
      if(t_next_step > t_target) {
#ifdef LOG
	logfile << " nop (next: " << t_next_step << ")";
#endif
	break;
      }
      else {
	//fire an ODE step
#ifdef LOG
	logfile << " step (next: " << t_next_step << ")" << endl;
#endif
	for(ode_table_t::iterator it = ode_delta.table.begin(); it != ode_delta.table.end(); ++it) {
#ifdef LOG
	  logfile << *(it->first) << " + "
		  << (*((it->second)[ode_delta.n_fired_steps])) << endl;
#endif
	  it->first->add_content_species(*((it->second)[ode_delta.n_fired_steps]));
	}
	++ode_delta.n_fired_steps;
      }
    }
#ifdef LOG
    logfile << endl;
#endif
  }

  void ode(ode_delta_t &ode_delta) {
    ode(ode_delta, ode_delta.t_end);
    ode_delta.clear();
  }
#endif
  


private:
  u01_vg_type *rng; //random number generator (uniform in 0-1)
#ifdef HYBRID
  //ode solver
  double **data;
#endif
#ifdef LOG
  ofstream &logfile;
#endif

  //utility: find a compartment in a vector of terms (could be removed?)
  int find_in_ptr_vector(vector<Compartment *> &v, Compartment *p) {
    for(unsigned int i = 0; i < v.size(); ++i)
      if(v[i] == p) return i;
    return -1;
  }

  multiplicityType my_round (double x) {
    return ((*rng)() < (x-floor(x))) ? (multiplicityType)ceil(x) : (multiplicityType)floor(x);
  }

  //stochastic walk on the matchset (and submatchset)
  pair<Rule*, Match *> get_mu(MatchSet &m, double d) {
    double s = 0;
    for(MatchSet::matchsetType::const_iterator it = m.matchset.begin(); it != m.matchset.end(); ++it) {
      for(unsigned int i = 0; i < it->second.size(); ++i) { //matches
	s += (it->second)[i]->rate;
	if(d < s)
	  //picked rule-and-match
	  return make_pair(it->first, (it->second)[i]);
      }
    }
    //never reached
    cerr << "Gillespie error" << endl;
    exit(1);
    return make_pair((Rule *)NULL, (Match *)NULL);
  }

  vector<SubMatchSet **> resolve_contents(SubMatchSet::contentsType &contents) {
    //cerr << "resolving content" << endl;
    multiplicityType sum = 0;
    SubMatchSet::contentsType::const_iterator it;
    for(it = contents.begin(); it != contents.end(); it++)
      for(vector<SubMatchSet **>::const_iterator jt = (*it)->begin(); jt != (*it)->end(); ++jt)
	sum += (**jt)->root.multiplicity;
    double d = (*rng)() * sum;
    //cerr << "sum: " << sum << "; d: " << d << " -> ";
    unsigned long long s = 0;
    for(it = contents.begin(); it != contents.end(); it++) {
      for(vector<SubMatchSet **>::const_iterator jt = (*it)->begin(); jt != (*it)->end(); ++jt)
	s += (**jt)->root.multiplicity;
      if(d <= double(s))
	break;
    }
    return **it;
  }

  void get_updates_(updatesType &updates, SubMatchSet &s, PCompartment *root) {
    //root update
    updates[root] = &(s.root);
    //other updates
    if(s.contents.size() > 0) {
      vector<SubMatchSet **> sms = resolve_contents(s.contents);
      for(vector<SubMatchSet **>::const_iterator it = sms.begin(); it != sms.end(); ++it)
	get_updates_(updates, ***it, (**it)->root.object);
    }
  }

  updatesType* get_updates(SubMatchSet &s, PCompartment *rule_lhs_root) {
    updatesType *utp;
    MYNEW(utp, updatesType);
    //updatesType &updates = *new updatesType;
    updatesType &updates = *utp;
    get_updates_(updates, s, rule_lhs_root);
    return &updates;
  }

  void do_updates(Rule &r, updatesType &updates,
#ifdef HYBRID
		  ode_delta_t &ode_delta,
#endif
		  Compartment &context) {

    vector<Compartment *> compartments_to_delete;
    instantiationType instantiations;

#ifdef HYBRID
    //ODE
    ode(ode_delta);
#endif
    
    //RESERVE
    PCompartment &rule_lhs_root(r.lhs);
    SubMatch &top_submatch(*updates[&rule_lhs_root]); //top level submatch
    Compartment &c(*top_submatch.term);
    c.update_delete((Compartment &)rule_lhs_root); //delete at top level
    //cout << "trunked term: " << *c << endl;
    instantiations[rule_lhs_root.content_variable_symbol] = make_pair(c.content_species, c.content_compartments); //reference top level X
    updates.erase(&rule_lhs_root); //remove the visited submatch
	
    //nested compartments
    for(updatesType::const_iterator it = updates.begin(); it != updates.end(); ++it) {
      //get a submatch and delete reactants
      PCompartment &pc(*it->first);
      SubMatch &s(*it->second);
      Compartment &c(*s.term);
      c.update_delete((Compartment &)pc);
      //reference variables
      instantiations[pc.content_variable_symbol] = make_pair(c.content_species, c.content_compartments);
      instantiations[pc.wrap_variable_symbol] = make_pair(c.wrap, (vector<Compartment *> *)NULL);
      //remove the compartment
      int position_to_delete = find_in_ptr_vector(*s.parent, s.term);
      compartments_to_delete.push_back(&c);
      s.parent->erase(s.parent->begin() + position_to_delete);
    }
  
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

  /*
    #ifdef HYBRID
    void limited_ode(vector<double> &k, Species &x, double tau) {
    vector<multiplicityType> &subject(x.concentrations);

    #ifdef LOG
    logfile << "(tau: " << tau << ") input ODE: ";
    for(unsigned int i=0; i<subject.size(); i++)
    logfile << subject[i] << " " << flush;
    logfile << endl;
    #endif
    
    //cast rates to array
    int nk(k.size());
    double *k_array = (double *)malloc(nk * sizeof(double));
    for(int i=0; i<nk; i++)
    k_array[i] = k[i];
    
    #ifdef LOG
    logfile << "rates: ";
    for(int i=0; i<nk; i++)
    logfile << k_array[i] << " ";
    logfile << endl;
    #endif
    
    //get concentrations (from multiplicities)
    for(int i=0; i<NPar; i++)
    data[0][i] = double(subject[i]);
    
    odeSolver(data, k_array, 0.0, tau, tau/(NPoint - 1));
    
    //set multiplicities (from concentrations)
    for (int i=0; i < NPar; i++)
    subject[i] = multiplicityType(round(data[NPoint-1][i])); //last state
    
    //clean
    free(k_array);
    
    #ifdef LOG
    logfile << "output ODE: ";
    for(unsigned int i=0; i<subject.size(); i++)
    logfile << subject[i] << " ";
    logfile << endl;
    #endif
    }
    #endif
  */
};
#endif
