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

#include "PSpecies.h"

/*
#ifdef SIMD
#include <xmmintrin.h> //SSE
#include <nmmintrin.h> //SSE4
#define ONES _mm_set1_epi32(1)
#define ZEROS _mm_set1_epi32(0)
#endif
*/

PSpecies::PSpecies(Species &copy) :
  Species(copy)
{
  for(unsigned int i=0; i<alphabet_size; i++)
    if(concentrations[i] > 0)
      active.push_back(i);
}

multiplicityType PSpecies::binomial(multiplicityType n, multiplicityType k) {
  multiplicityType r = 1;
  for(unsigned int d = 1; d <= k; d++) {
    r *= n--;
    r /= d;
  }
  return r;
}

multiplicityType PSpecies::match(Species &subject) {
  multiplicityType r = 1;
  for(unsigned int i=0; i<active.size(); i++) {
    multiplicityType n = subject[active[i]];
    multiplicityType k = concentrations[active[i]];
    if(k > n) return 0;
    r *= (k == 1)? n : binomial(n, (k < n/2)? k : n-k);
  }
  return r;
}

pair<multiplicityType, multiplicityType> PSpecies::match_biochemical(Species &subject) {
  multiplicityType res = 1;
  multiplicityType rarest = subject[active[0]]; //biochemical patterns have at least one species
  for(unsigned int i=0; i<active.size(); i++) {
    multiplicityType n = subject[active[i]];
    multiplicityType k = concentrations[active[i]];
    if(k > n) return pair<multiplicityType, multiplicityType>(0, 0);
    multiplicityType cnk = (k == 1)? n : binomial(n, (k < n/2)? k : n-k);
    res *= cnk;
    if(cnk < rarest) rarest = cnk;
  }
  return pair<multiplicityType, multiplicityType>(res, rarest);
}

/*** SSE version (fast only if k >> 1)
 //prepare registers
 __m128i N, *K, D, *R, MASK, TEST, X, OFFSET;

 K = (__m128i*) concentrations; //object
 multiplicityType r_chunk[4];
 R = (__m128i*) r_chunk;

 for (unsigned int i = 0; i < n_rounded; i+=4, K++) {
 //get 4 values for N
 N = _mm_set_epi32(subject[i+3], subject[i+2], subject[i+1], subject[i]);

 //K > N (instead of D <= K)
 if(!_mm_test_all_zeros(_mm_cmpgt_epi32(*K, N), ONES))
 {
 return 0;
 }

 //init D, R
 D = _mm_set1_epi32(1);
 *R = _mm_set1_epi32(1);

 //binomial for 4 couples
 while(true) {
 //find active Ns (and Ds)
 TEST = _mm_cmpgt_epi32(D, *K); //D > K (instead of D <= K)

 if(_mm_test_all_ones(TEST)) {
 break; //nothing to do
 }

 else {
 //loop step
 TEST = _mm_andnot_si128(TEST, ONES);

 //multiplicate Rs by active Ns
 OFFSET = _mm_sub_epi32(ONES, TEST); //mask inversion
 X = _mm_mullo_epi32(N, TEST); //value-or-0 mask
 MASK = _mm_add_epi32(X, OFFSET); //value-or-1 mask
 *R = _mm_mullo_epi32(*R, MASK); //R * N

 //divide Rs by active Ds
 X = _mm_mullo_epi32(D, TEST); //value-or-0 mask
 MASK = _mm_add_epi32(X, OFFSET); //value-or-1 mask
 *R =_mm_cvtps_epi32 (_mm_div_ps (_mm_cvtepi32_ps(*R), _mm_cvtepi32_ps(MASK))); //R / D (with casts)

 //next step for Ns and Ds
 N = _mm_sub_epi32(N, ONES); //N--
 D = _mm_add_epi32(D, ONES); //D++
 }
 }

 //collect results
 r *= r_chunk[0] * r_chunk[1] * r_chunk[2] * r_chunk[3];
 }
*/
