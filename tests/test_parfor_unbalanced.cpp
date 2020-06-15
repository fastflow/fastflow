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
/* Author: Massimo Torquati
 * Date:   January 2014
 * Modified: M. Aldinucci
 * Date: September 2014
 */
/*
 * Unbalanced parallel for computation
 *
 *
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <ff/utils.hpp>
#if !defined(USE_OPENMP) && !defined(USE_TBB)
#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>
#endif
#if defined(USE_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#endif

using namespace ff;

#define MAX_PARALLELISM        64
#define MY_RAND_MAX         32767

__thread unsigned long next = 1;

inline static double Random(void) {
    next = next * 1103515245 + 12345;
    return (((unsigned) (next/65536) % (MY_RAND_MAX+1)) / (MY_RAND_MAX +1.0));
}

inline static void SRandom(unsigned long seed) {
    next = seed;
}

inline static void compute(long end) {
    ticks_wait(end); //for(volatile long j=0;j<end;++j);
}

int main(int argc, char *argv[]) {
    long seed = 7919;
    long seqIter=100;
    long N = 10000;
    int nthreads = 3;
    if (argc>1) {
        if (argc<5) {
            printf(" use: %s seed seqIter maxN numthreads\n", argv[0]);
            printf("      %s 7919 100 100000 4\n", argv[0]);
            return -1;
        }
        seed=atol(argv[1]);
        seqIter=atol(argv[2]);
        N = atol(argv[3]);
        nthreads = atoi(argv[4]);
    }
    double dt=0.0;
    srandom(seed);
    SRandom(random());
#if !defined(USE_OPENMP) && !defined(USE_TBB) 
    ParallelFor ffpf(nthreads, (nthreads < ff_numCores())); 
    ffpf.disableScheduler(); // disable scheduler
#endif
#if defined(USE_TBB)
    tbb::task_scheduler_init init(nthreads);
    tbb::affinity_partitioner ap;
#endif

    for(int k=0; k<seqIter; ++k) { // external sequential loop
        unsigned long iter=0;
        long _N = (std::max)((long)(Random()*N),(long)MAX_PARALLELISM);
        
        //printf("n=%ld\n", _N);
        
        long *V;
        if (posix_memalign((void**)&V, 64, (_N+1)*sizeof(long)) != 0) 
            error("posix_memalign");
        
        for (long i=1; i<=_N;++i) {
            float c = ceil((float(_N/MAX_PARALLELISM))*powf(float(i),-1.1));	
            V[i] = (long)(c);
            iter += (long)(c);
            //printf("%ld ", V[i]);
        }
        //printf("_N=%ld iter=%ld\n", _N, iter);
        
        ffTime(START_TIME);
#if defined(USE_OPENMP)
#pragma omp parallel for schedule(dynamic,1) num_threads(nthreads)
        for(long i=1;i<=_N;++i) {
            compute(10000*V[i]);
        }
#elif defined(USE_TBB)
        tbb::parallel_for(tbb::blocked_range<long>(1, _N+1,1),
                          [&] (tbb::blocked_range<long> &r) {
                              for(long j=r.begin(); j!=r.end(); ++j) {
                                  compute(10000*V[j]);
                              }
                          },ap);
#else
        ffpf.parallel_for(1,_N+1,1,1,[V](const long i){compute(10000*V[i]);}, nthreads);
#endif
        ffTime(STOP_TIME);
        posix_memalign_free(V); // Win compability
        dt += ffTime(GET_TIME); 
    }
    printf("%d Time=%g (ms)\n", nthreads, dt);
    return 0;
}
