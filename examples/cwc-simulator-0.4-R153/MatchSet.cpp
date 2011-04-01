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

#include "MatchSet.h"
#include <cmath>
#ifdef HYBRID
#include "ode.h"
#endif
#include <iostream>


MatchSet::MatchSet(): rate_sum(0), rate_sum_deterministic(0) {};

MatchSet::~MatchSet() {
  for(matchsetType::const_iterator it = matchset.begin(); it != matchset.end(); ++it) {
    std::vector<Match *> &vs(*it->second);
    for(unsigned int i = 0; i < vs.size(); ++i)
      delete vs[i];
    delete &vs;
  }
  for(odeRatesType::const_iterator it = ode_rates.begin(); it != ode_rates.end(); it++) {
    delete it->second;
  }
}

void MatchSet::match(Model &m, double time) {
  for(unsigned int i = 0; i < m.rules.size(); ++i) {
    Rule &r = *m.rules[i];
    pair<vector<Match *> *, double> vs = match_rule_term(r, m.term, time);
    if(vs.first->size() > 0) {
      matchset[&r] = vs.first;
      rate_sum += vs.second;
    }
    else delete vs.first;
  }
}

pair<vector<Match *> *, double> MatchSet::match_rule_term(Rule &r, Compartment &vt, double time) {

  vector<Match *> *match_total = new vector<Match *>;
  double rate = 0;

  //on-level
  multiplicityType count_level;
  multiplicityType rarest = 0;
  SubMatchSet *sms = new SubMatchSet;

#ifdef HYBRID
  if(r.is_biochemical) {
    pair<multiplicityType, multiplicityType> sms_level = sms->submatch_biochemical(r.lhs, vt);
    count_level = sms_level.first;
    rarest = sms_level.second;
  }
  else {
    count_level = sms->submatch(r.lhs, vt);
  }

#else
  count_level = sms->submatch(r.lhs, vt);
#endif

  if(count_level > 0) {
    double match_rate;
    switch(r.semantics) {
    case 0:
      {
	//law-mass
	match_rate = count_level * r.parameters[0];
      }
      break;
    case 1:
      {
	//Michaelis-Menten (v,k)
	double &v(r.parameters[0]);
	double &k(r.parameters[1]);
	match_rate = v * count_level / (k + count_level);
      }
      break;
    case 2:
      {
	//Hill (v,k,n)
	double &v(r.parameters[0]);
	double &k(r.parameters[1]);
	double &n(r.parameters[2]);
	match_rate = v * pow(count_level, n) / (pow(k, n) + pow(count_level, n));
      }
      break;
    case 3:
      {
	//timed law-mass
	double &tl(r.parameters[1]);
	double &ts(r.parameters[2]);
	match_rate = count_level * r.parameters[0];
	match_rate /= (1 + exp((time-tl)/ts));
      }
    case 4:
      {
	//timed Michaelis-Menten
	double &v(r.parameters[0]);
	double &k(r.parameters[1]);
	double &tl(r.parameters[2]);
	double &ts(r.parameters[3]);
	match_rate = v * count_level / (k + count_level);
	match_rate /= (1 + exp((time-tl)/ts));
      }
    case 5:
      {
	//timed Hill
	double &v(r.parameters[0]);
	double &k(r.parameters[1]);
	double &n(r.parameters[2]);
	double &tl(r.parameters[3]);
	double &ts(r.parameters[4]);
	match_rate = v * pow(count_level, n) / (pow(k, n) + pow(count_level, n));
	match_rate /= (1 + exp((time-tl)/ts));
      }
      break;
    default:
      cerr << "undefined semantics: " << r.semantics << endl;
      match_rate = 0;
      break;
    }
    
    Match *match_level = new Match(*sms, match_rate, &vt, rarest);
    rate += match_rate;
    match_total->push_back(match_level);
  } else {
    // delete partially built submatch if match fails
    delete sms;
  }
	
  //in-depth
  vector<Compartment *> &subject_cc = *vt.content_compartments;
  for (unsigned int i=0; i<subject_cc.size(); i++) {
    Compartment &subject = *subject_cc[i];
    pair<vector<Match *> *, double> rec = match_rule_term(r, subject, time);
    vector<Match *> *match_rec = rec.first;
    //merge
    match_total->insert(match_total->end(), match_rec->begin(), match_rec->end());
    rate += rec.second;
    //clean up
    delete match_rec;
  }

  return pair<vector<Match *> *, double>(match_total, rate);
}


#ifdef HYBRID
void MatchSet::split(double rc, multiplicityType pc, unsigned int ode_size) {
  vector<MatchSet::matchsetType::iterator> rms_to_erase;
  for(MatchSet::matchsetType::iterator it = matchset.begin(); it != matchset.end(); ++it) { //rules
    Rule &rule = *it->first;
    vector<Match *> &matches(*it->second);
    if(rule.is_biochemical) {
      for(vector<Match *>::iterator jt = matches.begin(); jt != matches.end(); jt++) { //rule-matches
	Match &match = **jt;
	if(rc != RATE_CUTOFF_INFINITE && match.rate > rc && match.rarest > pc) {
	  //add to ODE matchset
	  odeRatesType::const_iterator oc = ode_rates.find(match.context);
	  if(oc != ode_rates.end())
	    for(unsigned int i=0; i<rule.parameters.size(); i++)
	      (*oc->second)[rule.ode_index + i] = rule.parameters[i];
	  else {
	    vector<double> *rates = new vector<double>(ode_size, 0);
	    for(unsigned int i=0; i<rule.parameters.size(); i++)
	      rates->at(rule.ode_index + i) = rule.parameters[i];
	    ode_rates[match.context] = rates;
	  }
	  //remove from stochastic matchset
	  rate_sum_deterministic += match.rate;
	  delete &match;
	  matches.erase(jt--);
	}
      }
      if(matches.size() == 0) {
	//erase the rule-matchset
	delete &matches;
	rms_to_erase.push_back(it);
      }
    }
  }
  rate_sum -= rate_sum_deterministic; //reduce the stochastic total rate
  for(unsigned int i=0; i<rms_to_erase.size(); i++)
    matchset.erase(rms_to_erase[i]);
  //TODO: check double implementation; very strange cases (i.e. x < x)
  if(matchset.size() == 0)
    rate_sum = 0;
}
#endif



std::ostream& operator<<(std::ostream &os, const MatchSet &m) {
  //stochastic
  os << "stochastic MS (sum of rates: " << std::fixed << m.rate_sum << ")" << endl;
  for(MatchSet::matchsetType::const_iterator it = m.matchset.begin(); it != m.matchset.end(); ++it) {
    //double mr = 0;
    os << *it->first << " : ";
    if(it->second->size() == 0) os << endl;
    else {
      for(unsigned int i = 0; i < it->second->size(); ++i) {
	//mr += (*it->second)[i]->rate;
	os << *(*it->second)[i];
	if(it->first->is_biochemical) os << " (biochemical: " << it->first->ode_index << ", rarest: " << (*it->second)[i]->rarest << ")"; 
	os << endl;
      }
    }
  }

  //deterministic
  os << "deterministic MS (sum of rates: " << std::fixed << m.rate_sum_deterministic << ")" << endl;
  for(MatchSet::odeRatesType::const_iterator it = m.ode_rates.begin(); it != m.ode_rates.end(); ++it) {
    os << *it->first << " : [ ";
    for(unsigned int i = 0; i < it->second->size(); ++i) {
      os << (*it->second)[i] << " ";
    }
    os << "]" << endl;
  }

  return os;
}
