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

#include "Match.h"
#include "SubMatchSet.h"

Match::Match(
	     SubMatchSet &submatchset,
	     double rate,
	     Compartment *context,
	     multiplicityType rarest
	     )
  : submatchset(submatchset), rate(rate), context(context), rarest(rarest) {}

Match::~Match() { delete &submatchset; }



std::ostream& operator<<(std::ostream &os, const Match &m) {
  //os << "match rate: " << std::fixed << m.rate;
    os << "<START match: M rate = " << std::fixed << m.rate << endl;
    os << m.submatchset << endl;
    os << "END match>" << endl;
  return os;
}
