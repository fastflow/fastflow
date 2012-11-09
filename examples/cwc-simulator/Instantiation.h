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

#ifndef INSTANTIATION
#define INSTANTIATION

//#include "Term.h"
#include "definitions.h"
//#include "Compartment.h"
//#include "Species.h"
#include <map>
#include <string>
#include <vector>
using namespace std;

class Compartment;
class Species;

//typedef map<variable_symbol, pair<Species *, vector<Compartment *> *> > instantiationType;
typedef map<variable_adress, pair<Species *, vector<Compartment *> *> > instantiationType;

#endif
