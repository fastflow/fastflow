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
 */
/*
 * Unbalanced parallel for computation
 *
 *
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ff/utils.hpp>
#if !defined(USE_OPENMP)
#include <ff/parallel_for.hpp>
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
  for(volatile long j=0;j<end;++j);
}

int main(int argc, char *argv[]) {
    if (argc<5) {
        printf(" use: %s seed seqIter maxN numthreads [chunk=1]\n", argv[0]);
        printf("      %s 7919 100 100000 4\n", argv[0]);
        return -1;
    }
    long chunk = 1;
    if (argc == 6) chunk = atoi(argv[5]);
    srandom(atol(argv[1]));
    long seqIter=atol(argv[2]);
    long N = atol(argv[3]);
    int nthreads = atoi(argv[4]);
    double dt=0.0;

    SRandom(random());

    FF_PARFOR_INIT(pf, nthreads);

    for(int k=0; k<seqIter; ++k) { // external sequential loop
	unsigned long iter=0;
	long _N = std::max((long)(Random()*N),(long)MAX_PARALLELISM);
	
	//printf("n=%ld\n", _N);

	long *V;
	posix_memalign((void**)&V, 64, (_N+1)*sizeof(long));

	for (long i=1; i<=_N;++i) {
	    double c = ceil((_N/MAX_PARALLELISM)*powf(i,-1.1));	
	    V[i] = (long)(c);
	    iter += (long)(c);
	    //printf("%ld\n", V[i]);
	}
	//printf("_N=%ld iter=%ld\n", _N, iter);

	ffTime(START_TIME);
#if defined(USE_OPENMP)
#pragma omp parallel for schedule(runtime) num_threads(nthreads)
	for(long i=1;i<=_N;++i) {
	  compute(10000*V[i]);
	}
#else
	FF_PARFOR_START(pf, i,1,_N,1, chunk, nthreads) {
	  compute(10000*V[i]);
	} FF_PARFOR_STOP(pf);
#endif
	ffTime(STOP_TIME);
	free(V);
	dt += ffTime(GET_TIME); 
    }
    printf("%d Time=%g (ms)\n", nthreads, dt);

    FF_PARFOR_DONE(pf);
    return 0;
}
