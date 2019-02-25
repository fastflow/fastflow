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
 * Simple pool pattern test working on stream:
 *
 *          --------        -------        ----------
 *         | PopGen | ---> | Pool  | ---> | PopPrint |
 *          --------        -------        ----------
 */


#include <cstdio>
#include <vector>
#include <cstdlib>
#include <ff/ff.hpp>
#include <ff/poolEvolution.hpp>

using namespace ff;

#define STREAMLENGTH  10

#define MAXMUTATIONS  4
#define K             2

static int debug =0;

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

// selects all odd elements that have a number of mutation less than MAXMUTATIONS
void selection(ParallelForReduce<Element> &, 
               std::vector<Element> &P, 
               std::vector<Element> &output,poolEvolution<Element>::envT&) {
    for(size_t i=0;i<P.size()/2;++i)
        output.push_back(P[i]);
}

const Element& evolution(Element & individual,const poolEvolution<Element>::envT&,const int) {
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
        auto r = random() % output.size();  // MA Changed from P.size() to output.size()
        output.erase(output.begin()+r);
    }
}


template<typename T>
void printPopulation(std::vector<T> &P) {
    printf("[ ");
    for (size_t i=0;i<P.size(); ++i)
        printf(" (%zu,%zu) ", (size_t)P[i].number, (size_t)P[i].nmutations);
    printf("  ]\n");
}

template<typename T>
void buildPopulation(std::vector<T> &P, size_t size) {
    for (size_t i=0;i<size; ++i){
        Element E(random());
        P.push_back(E);
    }
}


struct BuildPop: ff_node {
    BuildPop(size_t maxsize):maxsize(maxsize) {
    }
    int svc_init() {
        srandom(10);
        return 0;
    }
    void *svc(void*) {
        for(int i=0;i<STREAMLENGTH;++i) {
            std::vector<Element> *V= new std::vector<Element>;
            buildPopulation(*V, random()%maxsize);
#if 0
            if (debug) {
                printf("Initial population:\n");
                printPopulation(*V);
            }
#endif
            ff_send_out(V);
        }
        return NULL;
    }    
    size_t maxsize;
};

struct PrintPop: ff_node {
    void *svc(void *task) {
        std::vector<Element> *V = (std::vector<Element>*)task;
        if (debug) {
            
            printPopulation(*V); 
        }        
        delete V;
        return GO_ON;
    }

};


int main(int argc, char* argv[]) {
    int nw       = 3;
    long maxsize = 500;

    if (argc>1) {
        if (argc < 3) {
            printf("use: %s evolution_par_degree size [debug=0|1]\n", argv[0]);
            return -1;
        }
        nw       =atoi(argv[1]);
        maxsize  =atol(argv[2]); 
        if (argc==4) debug=atoi(argv[3]);
    }
    
    BuildPop bp(maxsize);
    poolEvolution<Element> pool(nw, selection,evolution,filter,termination);
    PrintPop pp;

    ff_pipeline pipe;
    pipe.add_stage(&bp);
    pipe.add_stage(&pool);
    pipe.add_stage(&pp);

    ffTime(START_TIME);
    pipe.run_and_wait_end();
    ffTime(STOP_TIME);

    printf("Time: %g\n", ffTime(GET_TIME));
    return 0;
}



