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

#include <cstdio>
#include <vector>
#include <cstdlib>

#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>
#include <ff/poolEvolution.hpp>

using namespace ff;

long nwS=0;
long nwF=0;
long nwE=0;


struct Env_t {
    Env_t(long avg,long iter):avg(avg),iter(iter) {}
    long avg;
    long iter;
};

void buildPopulation(std::vector<long> &P, size_t size) {
    for (size_t i=0;i<size; ++i){
        P.push_back(random() % 10000000);
    }
}


long fitness(ParallelForReduce<long> & pfr, std::vector<long> &P, long nw) {
    long sum = 0;

    auto Fsum = [](long& v, const long elem) { v += elem; };
    pfr.parallel_reduce(sum, 0L,
                        0L, (long)(P.size()), 1L, 0,  // default static scheduling
                        [&](const long i, long& sum) {
                            sum += P[i];
                        }, 
                        Fsum,
                        nw); 

    return sum/P.size();
}

void selection(ParallelForReduce<long> & pfr, std::vector<long> &P, std::vector<long> &buffer,Env_t &env) {

    std::vector<std::vector<long> >  bufferPool(nwS);
    env.avg = fitness(pfr, P, nwS);

    auto S = [&](const long start, const long stop, const int thread_id) {

        // losing time
        ticks_wait(stop-start); //for(volatile long j=0;j<(stop-start);++j);  

        for(long j=start; j<stop; ++j) {
            if (P[j] > env.avg) bufferPool[thread_id].push_back(P[j]);
        }
    };
    pfr.parallel_for_idx(0,P.size(),1, 0,S,nwS);  // default static scheduling
    buffer.clear();

    for(size_t i=0;i<(size_t)nwS;++i)
        buffer.insert( buffer.end(), bufferPool[i].begin(), bufferPool[i].end());

}

const long &evolution(long &element, const Env_t&,const int) {
    ticks_wait(element%5000);
    //for(volatile long j=0;j<element%5000;++j);   // lose time
    
    if (element & 0x1) element += 1;
    else element -=1;
    return element;
}

void filter(ParallelForReduce<long> & pfr, std::vector<long> &, std::vector<long> &buffer,Env_t &env) {
    env.avg = fitness(pfr, buffer, nwF);
    env.iter +=1;

    pfr.parallel_for(0L, (long)(buffer.size()),1L, 8, // dynamic scheduling with grain 8
                     [&](const long i) {
                         ticks_wait(buffer[i]%5000);
                         //for(volatile long j=0;j<buffer[i]%5000;++j);   // lose time
                     }, nwF );
    
}

bool termination(const std::vector<long> &P, Env_t &env) {
    
    if (P.size()<=2 || env.avg < 100 || env.iter > 50) return true;

    return false;
}




int main(int argc, char* argv[]) {
    long size=0;
    if (argc>1) {
        if (argc < 5) {
            printf("use: %s nwS nwE nwF size \n", argv[0]);
            return -1;
        }
        nwS    = atoi(argv[1]);
        nwE    = atoi(argv[2]);
        nwF    = atoi(argv[3]);
        size   = atol(argv[4]); 
    }

    srandom(10);
    
    std::vector<long> P;
    std::vector<long> buffer;
    buildPopulation(P, size);

    Env_t env(1000000,0);

    poolEvolution<long, Env_t> pool((std::max)(nwF, (std::max)(nwS,nwE)), P, selection,evolution,filter,termination, env);
    pool.run_and_wait_end();

    
    
    printf("final avg = %ld (iter i=%ld)\n", pool.getEnv().avg, pool.getEnv().iter);

    return 0;
}



