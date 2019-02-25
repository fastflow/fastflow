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
/*
 * Simple pool pattern test.
 *
 *
 */


#include <cstdio>
#include <vector>
#include <cstdlib>
#include <ff/ff.hpp>
#include <ff/poolEvolution.hpp>

using namespace ff;

#define MAXMUTATIONS  4
#define K             2


struct Element {
    Element(size_t n=0): number(n),nmutations(0) {}
    size_t number;
    size_t nmutations;
};


// if we have at least an odd element, than we go on
bool termination(const std::vector<Element> &P, poolEvolution<Element>::envT&) {
    for(size_t i=0;i<P.size(); ++i)
        if (P[i].nmutations < MAXMUTATIONS && P[i].number & 0x1) return false;
    return true;
}

// selects all odd elements that have a number of mutation less than MAXMUTATIONS
void selection(ParallelForReduce<Element> &, 
               std::vector<Element> &P, 
               std::vector<Element> &output,poolEvolution<Element>::envT&) {
    for(size_t i=0;i<P.size()/2;++i)
        output.push_back(P[i]);
}

const Element& evolution(Element & individual, const poolEvolution<Element>::envT&,const int) {
    individual.number += decltype(individual.number)(individual.number/2);
    individual.nmutations +=1;
    return individual;
}

// remove at most K elements randomly
void filter(ParallelForReduce<Element> &, 
            std::vector<Element> &P, 
            std::vector<Element> &output,poolEvolution<Element>::envT&) {
    
    if (P.size()<K) { output.clear(); return; }
    output.clear();
    output.insert(output.begin(),P.begin(), P.end());

    for(size_t i=0;i<K;++i) {
        auto r = random() % (output.size()); // MA Changed from P.size() to output.size()
        //std::cout << "Erease " << r << " Size " << output.size() << "\n";
        output.erase(output.begin()+r);
        // std::cout << "Done " << r << " Size " << output.size() << "\n";
    }
}


template<typename T>
void printPopulation(std::vector<T> &P) {
    printf("[ ");
    for (size_t i=0;i<P.size(); ++i)
        printf(" (%ld,%ld) ", (size_t)P[i].number, (size_t)P[i].nmutations);
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
    int nw    = 3;
    long size = 500;
    int debug = 0;
    if (argc>1) {
        if (argc < 3) {
            printf("use: %s evolution_par_degree size [debug=0|1]\n", argv[0]);
            return -1;
        }
        nw    =atoi(argv[1]);
        size  =atol(argv[2]); 
        if (argc==4) debug=atoi(argv[3]);
    }
    
    srandom(10);
    
    std::vector<Element> P;
    buildPopulation(P, size);
    
    if (debug) {
        printf("Initial population:\n");
        printPopulation(P);
    }
    
    ffTime(START_TIME);
    poolEvolution<Element> pool(nw, P,selection,evolution,filter,termination);
    pool.run_and_wait_end();
    ffTime(STOP_TIME);

    if (debug) {
        printf("Final population:\n");
        printPopulation(P);
    }

    printf("Time: %g\n", ffTime(GET_TIME));
    return 0;
}



