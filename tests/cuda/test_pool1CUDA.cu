/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

#if !defined(FF_CUDA)
#define FF_CUDA
#endif


#include <cstdio>
#include <vector>
#include <cstdlib>
#include <ff/stencilReduceCUDA.hpp>
#include <ff/poolEvolutionCUDA.hpp>

using namespace ff;

#define MAXMUTATIONS  4
#define KELEM             2


struct Element {
    Element(size_t n): number(n),nmutations(0) {}

    __host__ __device__ Element() {}

    __host__ __device__ Element(const Element& E):number(E.number),nmutations(E.nmutations) {}
    __device__ Element(volatile const Element& E):number(E.number),nmutations(E.nmutations) {}
    
    __device__ volatile Element& operator=(volatile Element& E) volatile {
        number = E.number;
        nmutations = E.nmutations;
        return *this;
    }
    __host__ __device__ Element& operator=(const Element& E) {
        number = E.number;
        nmutations = E.nmutations;
        return *this;
    }
    size_t number;
    size_t nmutations;
};


FFMAPFUNC(evolMap, Element, individual,
          individual.number += (size_t)(individual.number/2);
          individual.nmutations +=1;
          return individual;
);


// if we have at least an odd element, than we go on
bool termination(const std::vector<Element> &P) {
    for(size_t i=0;i<P.size(); ++i)
        if (P[i].nmutations < MAXMUTATIONS && P[i].number & 0x1) return false;
    return true;
}

void selection(std::vector<Element>::const_iterator P_start, 
               std::vector<Element>::const_iterator P_stop, 
               std::vector<Element> &output) {
    size_t size = P_stop-P_start;
    for(size_t i=0;i<(size/2);++i)
        output.push_back(P_start[i]);     
}


// remove at most KELEM elements randomly
void filter(std::vector<Element>::const_iterator P_start, 
            std::vector<Element>::const_iterator P_stop, 
            std::vector<Element> & output) {
    
    size_t size = P_stop-P_start;
    if (size<KELEM) { output.clear(); return; }
    output.insert(output.begin(),P_start,P_stop);

    for(size_t i=0;i<KELEM;++i) {
        size_t r = random() % size;
        output.erase(output.begin()+r);
    }
}

template<typename T>
void printPopulation(std::vector<T> &P) {
    printf("[ ");
    for (size_t i=0;i<P.size(); ++i)
        printf(" (%u,%u) ", (size_t)P[i].number, (size_t)P[i].nmutations);
    printf("  ]\n");
}

template<typename T>
void buildPopulation(std::vector<T> &P, size_t size) {
    for (size_t i=0;i<size; ++i){
        Element E(random());
        P.push_back(E);
    }
}


int main(int argc, char* argv[]) {
    long size = 500;
    int debug = 0;
    if (argc>1) {
        if (argc < 3) {
            printf("use: %s size debug=0|1\n", argv[0]);
            return -1;
        }
        size  =atol(argv[1]); 
        debug=atoi(argv[2]);
    }
    
    srandom(10);
    
    std::vector<Element> P;
    buildPopulation(P, size);
    
    if (debug) {
        printf("Initial population:\n");
        printPopulation(P);
    }
    
    ffTime(START_TIME);
    poolEvolutionCUDA<Element, evolMap> pool(P,selection,filter,termination);
    pool.run_and_wait_end();
    ffTime(STOP_TIME);
    
    if (debug) {
        printf("Final population:\n");
        printPopulation(P);
    }
    
    printf("Time: %g\n", ffTime(GET_TIME));
    return 0;
}



