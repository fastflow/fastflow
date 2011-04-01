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

#ifndef MATCHSET
#define MATCHSET
#include "Match.h"
#include "Model.h"
#include "Rule.h"
#include <map>
#include <vector>
#include <ostream>
using namespace std;

/**
   matches of a ruleset against a compartment
*/
class MatchSet {

 public:
  /**
     constructor (empty matchset)
  */
  MatchSet();
  ~MatchSet();

  typedef map<Rule *, vector<Match *> *> matchsetType;
  typedef map<Compartment *, vector<double> *> odeRatesType;

  /**
     association between rules and their matches (one for each context)
  */
  matchsetType matchset;

  /**
     total rate of the stochastic matchset
  */
  double rate_sum;

  /**
     activation-array for the ODE
  */
  odeRatesType ode_rates;

  /**
     total rate of the deterministic matchset
  */
  double rate_sum_deterministic;

  /**
     build the matchset over a model
  */
  void match(Model &model, double time);

  /**
     rule matching.
     Computes the matches of a rule against a compartment, updating the matchset
     @param rule the object rule
     @param subject the subject compartment
     @return matches and total rate
  */
  pair<vector<Match *> *, double> match_rule_term(Rule &rule, Compartment &subject, double time);

#ifdef HYBRID
  /**
     cutoff-based ODE-stochastic split
     @param rc rate cutoff
     @param pc population cutoff
     @param ode_size the number of equations in the ODE system
  */
  void split(double rc, multiplicityType pc, unsigned int ode_size);
#endif

  friend std::ostream& operator<<(std::ostream &, const MatchSet &);
};
#endif
