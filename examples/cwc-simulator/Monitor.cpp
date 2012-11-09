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

#include "Monitor.h"
#include "PCompartment.h"
#include "utils.h"
#include "Compartment.h"

Monitor::Monitor (
		  string &t,
		  Species *s,
		  vector<PCompartment *> *p,
		  type_adress tp
		  )
  :
  title(t)
  //, pattern(new PCompartment(s, p, *new string, tp))
{
  //string *strp;
  //MYNEW(strp, string);
  //MYNEW(pattern, PCompartment, s, p, *new string, tp);
  MYNEW(pattern, PCompartment, s, p, VARIABLE_ADRESS_RESERVED, tp);
}

Monitor::Monitor(Monitor &copy) :
  title(*new string(copy.title))
  /*, pattern(new PCompartment(*copy.pattern))*/
{
  MYNEW(pattern, PCompartment, *copy.pattern);
}

Monitor::~Monitor() {
  delete &title;
  delete pattern;
}

std::ostream& operator<<(std::ostream &os, const Monitor &m) {
  os << "\"" << m.title << "\": ";
  return os << m.pattern;
}
