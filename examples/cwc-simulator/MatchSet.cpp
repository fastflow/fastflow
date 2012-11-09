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
#include "SubMatchSet.h"
#include <cmath>
#ifdef HYBRID
#include "ode.h"
#endif
#include <iostream>
#include "Model.h"
#include "Match.h"
#include "Rule.h"
#include "utils.h"


MatchSet::MatchSet(): rate_sum(0), rate_sum_deterministic(0)
{
#ifdef HYBRID
  ode_only = false;
#endif
};

void MatchSet::init(Model &m
#ifdef LOG
		    , ostream *logfile
#endif
		    ) {
  for(unsigned int i = 0; i < m.rules.size(); ++i) {
    matchset[m.rules[i]] = vector<Match *>();
    matchset[m.rules[i]].reserve(10);
  }
#ifdef LOG
  this->logfile = logfile;
#endif
}

void MatchSet::clear() {
  for(MatchSet::matchsetType::iterator it = matchset.begin(); it != matchset.end(); ++it) {
    for(vector<Match *>::const_iterator jt = (it->second).begin(); jt != (it->second).end(); ++jt)
      delete *jt;
    (it->second).clear();
  }
  rate_sum = 0;

#ifdef HYBRID
  for(MatchSet::odeRatesType::iterator it = ode_rates.begin(); it != ode_rates.end(); ++it)
    delete it->second;
  ode_rates.clear();
  rate_sum_deterministic = 0;
  ode_only = false;
#endif
}

void MatchSet::match(Model &m, double time) {
  for(unsigned int i = 0; i < m.rules.size(); ++i) {
    /*
      #ifdef LOG
      *logfile << "will match rule " << i << ": {addr:" << m.rules[i] << "}" << *m.rules[i] << endl;
      #endif
    */
    Rule *r = m.rules[i];
    rate_sum += match_rule_term(*r, m.term, time, matchset[r]);
  }
}

double MatchSet::match_rule_term(Rule &r, Compartment &vt, double time, vector<Match *> &matches) {

  double rate = 0;

  //on-level

  //compute occurrences of the whole pattern
  multiplicityType count_level = 0;
  multiplicityType rarest = 0;
  SubMatchSet *sms;
  //sms= new SubMatchSet;
  MYNEW(sms, SubMatchSet);

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

    //match found: rating
    double match_rate;
    switch(r.semantics) {
    
    case 0:
      //mass-action
      {
	match_rate = count_level * r.parameters[0];
      }
      break;
    
    case 1:
      //Michaelis-Menten (v,k)
      {
	double &v(r.parameters[0]);
	double &k(r.parameters[1]);
	match_rate = v * count_level / (k + count_level);
      }
      break;
    
    case 2:
      //Hill (v,k,n)
      {
	double &v(r.parameters[0]);
	double &k(r.parameters[1]);
	double &n(r.parameters[2]);
	//match_rate = v * pow(count_level, n) / (pow(k, n) + pow(count_level, n));
	match_rate = v * pow(k, n) / (pow(k, n) + pow(count_level, n));
      }
      break;
    
    case 3:
      //timed law-mass
      {
	double &tl(r.parameters[1]);
	double &ts(r.parameters[2]);
	match_rate = count_level * r.parameters[0];
	match_rate /= (1 + exp((time-tl)/ts));
      }
      break;
    
    case 4:
      //timed Michaelis-Menten
      {
	double &v(r.parameters[0]);
	double &k(r.parameters[1]);
	double &tl(r.parameters[2]);
	double &ts(r.parameters[3]);
	match_rate = v * count_level / (k + count_level);
	match_rate /= (1 + exp((time-tl)/ts));
      }
      break;
    
    case 5:
      //timed Hill
      {
	double &v(r.parameters[0]);
	double &k(r.parameters[1]);
	double &n(r.parameters[2]);
	double &tl(r.parameters[3]);
	double &ts(r.parameters[4]);
	match_rate = v * pow(k, n) / (pow(k, n) + pow(count_level, n));
	//match_rate /= (1 + exp((-time-tl)/ts));//Modificato il timing in senso di attivazione!! (segno meno davanti a "time")
	//if((time>tl&&time<2*tl)||(time>3*tl)) match_rate *=ts;
	if ( fmod((double)time/tl,2)>1 ) match_rate*=ts;
      }
      break;
    
    case 6:
      //functional dummy
      {
#ifdef LOG
	* logfile << "functional match found on context: " << vt << " {" << &vt << "}" << endl;
#endif
	//compute occurrences of specific species
	vector<multiplicityType> occs(r.species_occs.size(), 0);
	for(unsigned int i=0; i<occs.size(); ++i)
	  occs[i] = (*(vt.content_species))[r.species_occs[i]];
	double &k(r.parameters[0]);
	double &h0(r.parameters[1]);
	match_rate = k * (h0 - occs[0]);
#ifdef LOG
	* logfile << "rate: k * (h0 - occs[0]) = "
		  << k << "* (" << h0 << " - " << occs[0] << ") = "
		  << match_rate << endl;
#endif
      }
      break;
      
    case 7: // Jchannel =(1-g)*(A*ip3^4/(ip3+k)^4+l)*Cas
      {
	double k=r.parameters[0];
	double l=r.parameters[1];
	double a=r.parameters[2];

	//compute occurrences of specific species
	vector<multiplicityType> occs(r.species_occs.size(), 0);
	for(unsigned int i=0; i<occs.size(); ++i)
	  occs[i] = (*(vt.content_species))[r.species_occs[i]];

	multiplicityType ip3=occs[0]; // <----- RISPETTARE QUEST'ORDINE nel codice cwc !!! SALVATORE
	multiplicityType g=occs[1];
      
	match_rate=ip3/200.0;
	match_rate=match_rate/(match_rate+k);
	match_rate*=match_rate;
	match_rate*=match_rate;
	match_rate=(match_rate*a+l)*count_level*(1.0-g/100.0);
      }
      break;

    case 8: // JPump =B*(CaI*0.01)^2/((CaI*0.01)^2+k^2);
      {
	double k=r.parameters[0];
	double b=r.parameters[1];

	match_rate=count_level*count_level*0.0001;
	match_rate=b*match_rate/(match_rate+k*k);
      }
      break;
    
    case 9: // kPLC =C*(1-k3/(CaI*0.01+k)*1/(1+R)) *100;
      {
	double k=r.parameters[0];
	double R=r.parameters[1];
	double c=r.parameters[2];
      
	match_rate=c*(1.0-k/(count_level*0.01+k)/(1.0+R))*100.0;
      }
      break;
    
    case 10: // kPhosphatase =D*IP3*0.5 *100;
      {
	double d=r.parameters[0];
      
	match_rate=d*count_level*0.5;
      }
      break;

    case 11: // inhibition_parameter1=global_par_E*(CaI*0.01)^4*(1-g/100)) *100;
      {
	double e=r.parameters[0];

	//compute occurrences of specific species
	vector<multiplicityType> occs(r.species_occs.size(), 0);
	for(unsigned int i=0; i<occs.size(); ++i)
	  occs[i] = (*(vt.content_species))[r.species_occs[i]];

	multiplicityType g=occs[0]; // <----- RISPETTARE QUEST'ORDINE nel codice cwc !!! SALVATORE

	match_rate=count_level*count_level*0.0001;
	match_rate*=match_rate;
	match_rate=e*match_rate*(1.0-g/100.0) *100.0;
      }
      break;
    
    case 12: // inibition_parameter2 =F *100; E' una mass-action fa solo il riscalamento.
      {
	double f=r.parameters[0];
	match_rate=f*100.0;
      }
      break;
    
    default:
      cerr << "undefined semantics: " << r.semantics << endl;
      match_rate = 0;
    }

    Match *match;
    //match = new Match(*sms, match_rate, &vt, rarest);
    MYNEW(match, Match, *sms, match_rate, &vt, rarest);
    matches.push_back(match);
    rate += match_rate;
  }
  else
    //match not found: delete partial submatch
    delete sms;
	
  //in-depth
  vector<Compartment *> &subject_cc = *vt.content_compartments;
  for (unsigned int i=0; i<subject_cc.size(); i++)
    rate += match_rule_term(r, *subject_cc[i], time, matches);

  return rate;
}


#ifdef HYBRID
void MatchSet::split(double rc, multiplicityType pc, unsigned int ode_size) {
  //cerr << "splitting:" << endl << (*this) << endl;
  vector<MatchSet::matchsetType::iterator> rms_to_erase;
  for(MatchSet::matchsetType::iterator it = matchset.begin(); it != matchset.end(); ++it) { //rules
    Rule &rule = *it->first;
    vector<Match *> &matches(it->second);
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
	//delete &matches;
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
  
  //cerr << "result:" << endl << (*this) << endl;
}
#endif



std::ostream& operator<<(std::ostream &os, const MatchSet &m) {
  os << "< start MS" << endl;
  os << "+ stochastic" << endl;
  os << "size: " << m.matchset.size() << endl;
  os << "total: " << m.rate_sum << endl;
  /*
    unsigned int i=0;
    for(MatchSet::matchsetType::const_iterator it = m.matchset.begin(); it != m.matchset.end(); ++it) {
    os << "rule #" << (i++) << ": " << *(it->first) << endl;
    for(unsigned int i=0; i<it->second.size(); ++i) {
    os << "match #" << i << ": {" << it->second[i] << "} rate: " << it->second[i]->rate << endl;
    //os << *(it->second[i]);
    }

    }
  */
  //os << "end MS >" << endl;
  //return os;

  /*
  //stochastic
  os << "stochastic MS (sum of rates: " << std::fixed << m.rate_sum << ")" << endl;
  for(MatchSet::matchsetType::const_iterator it = m.matchset.begin(); it != m.matchset.end(); ++it) {
  //double mr = 0;
  if(it->second.size() > 0) {
  os << *it->first << endl; //rule
  os << "-\n";
  for(unsigned int i = 0; i < it->second.size(); ++i) {
  os << *(it->second)[i] << endl; //i-th match
  if(it->first->is_biochemical)
  os << "(biochemical: " << it->first->ode_index << ", rarest: " << (it->second)[i]->rarest << ")"; 
  os << endl;
  }
  }
  }
  */

#ifdef HYBRID
  //deterministic
  os << "+ deterministic" << endl;
  os << "size: " << m.ode_rates.size() << endl;
  os << "total: " << m.rate_sum_deterministic << endl;
  /*
    for(MatchSet::odeRatesType::const_iterator it = m.ode_rates.begin(); it != m.ode_rates.end(); ++it) {
    os << "{" << it->first << "} " << *(it->first) << " : [ ";
    for(unsigned int i = 0; i < it->second->size(); ++i)
    os << (*it->second)[i] << " ";
    os << "]" << endl;
    }
  */
#endif
  os << "end MS >" << endl;

  return os;
}
