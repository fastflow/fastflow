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

#ifndef SPECIES
#define SPECIES

#include "definitions.h"
#include <vector>

#include <ostream>
using namespace std;

class Species {
 public:
  vector<multiplicityType> &concentrations;

  Species(unsigned int alphabet_size);
  Species(Species &);
  ~Species();

  multiplicityType &operator[](symbol_adress);

  //multiplicityType match(Species &subject);
  void add(symbol_adress atom, multiplicityType n);
  void update_add(Species &object);
  void update_delete(Species &object);
  void diff(Species &, Species &);

  friend ostream& operator<<(ostream &, Species &);

 protected:
  const unsigned int alphabet_size;
  //multiplicityType *concentrations;
#ifdef SIMD
  const unsigned int n_rounded;
#endif
};
#endif

