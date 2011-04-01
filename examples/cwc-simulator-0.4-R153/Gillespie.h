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
#include "random.h"

#ifndef HYBRID
//pure stochastic semantics
void stochastic_gillespie(MatchSet &, double &, u01_vg_type &);

#else
//hybrid semantics
void hybrid_gillespie(MatchSet &, double &, u01_vg_type &);
void retry(MatchSet &, double &time, u01_vg_type &);
#endif
