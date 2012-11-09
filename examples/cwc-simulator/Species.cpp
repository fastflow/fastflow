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

#include "Species.h"
#include <iostream>

#ifdef SIMD
#include <xmmintrin.h> //SSE
#include <nmmintrin.h> //SSE4
#define ONES _mm_set1_epi32(1)
#define ZEROS _mm_set1_epi32(0)
#endif

#ifdef SIMD
Species::Species(unsigned int n) :
  concentrations((multiplicityType *) _mm_malloc(n_rounded * sizeof(multiplicityType), 16)), //16-byte aligned
  alphabet_size(n),
  n_rounded(n + n % 4)
{
  __m128i *cr = (__m128i *)concentrations;
  for(unsigned int i=0; i<n_rounded; i+=4, cr++)
    *cr = ZEROS;
}
#else
Species::Species(unsigned int n) :
  concentrations(*new vector<multiplicityType>(n, 0)),
  alphabet_size(n) {}
#endif
  
#ifdef SIMD
Species::Species(Species &copy) :
  Species(copy.alphabet_size)
{
  concentrations = (multiplicityType *) _mm_malloc(n_rounded * sizeof(multiplicityType), 16); //16-byte aligned
  __m128i *cr = (__m128i *)concentrations;
  for(unsigned int i=0; i<n_rounded; i+=4, cr++)
    *cr = _mm_set_epi32(copy[i+3], copy[i+2], copy[i+1], copy[i]);
}
#else
Species::Species(Species &copy) :
  concentrations(*new vector<multiplicityType>(copy.concentrations)),
  alphabet_size(copy.alphabet_size) {}
#endif


Species::~Species() {
#ifndef SIMD
  delete &concentrations;
#else
  _mm_free(concentrations);
#endif
}

multiplicityType &Species::operator[](symbol_adress atom) {
  return concentrations[atom];
}

void Species::add(symbol_adress atom, multiplicityType n) {
  concentrations[atom] += n;
}

void Species::update_add(Species &object) {
#ifndef SIMD
  for(unsigned int i=0; i<alphabet_size; i++)
    concentrations[i] += object[i];
#else
  __m128i* sub = (__m128i*) concentrations;
  __m128i* obj = (__m128i*) object.concentrations;
  for (unsigned int i = 0; i < n_rounded; i+=4, obj++, sub++)
    {
      //operations on a chunk (4 packed 32-bit floats)
      *sub = _mm_add_epi32(*obj, *sub); // *obj + *sub
    }
#endif
}

void Species::update_delete(Species &object) {
#ifndef SIMD
  for(unsigned int i=0; i<alphabet_size; i++)
    concentrations[i] -= object[i];
#else
  __m128i* sub = (__m128i*) concentrations;
  __m128i* obj = (__m128i*) object.concentrations;
  for (unsigned int i = 0; i < n_rounded; i+=4, obj++, sub++)
    {
      //operations on a chunk (4 packed 32-bit floats)
      *sub = _mm_sub_epi32(*sub, *obj); // *sub - *obj
    }
#endif
}

void Species::diff(Species &object, Species &out) {
  for(unsigned int i=0; i<alphabet_size; ++i)
    out.concentrations[i] = concentrations[i] - object[i];
}

ostream& operator<<(ostream &os, Species &s) {
  //return os << "SPECIES";
  for(unsigned int i=0; i<s.alphabet_size; i++) {
    multiplicityType x = s[i];
    if(x != 0)
      os << x << "*" << i << " ";
  }
  return os;
}
