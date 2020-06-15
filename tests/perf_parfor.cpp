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
 * Date:   June 2013
 */
/* Very simple performance test for the FF_PARFOR
 *
 */

#include <cstdlib>
#if defined(USE_OPENMP)
#include <omp.h>
#endif

#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>

#if defined(USE_TBB)
#include <tbb/tbb.h>
#include <tbb/tbb_thread.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#endif



/*
 * This random generators are implementing 
 * by following POSIX.1-2001 directives.
 */

#define SIM_RAND_MAX         32767
__thread unsigned long next = 0;

inline static long simRandom(void) {
    next = next * 1103515245 + 12345;
    return((unsigned)(next/65536) % 32768);
}

inline static void simSRandom(unsigned long seed) {
    next = seed;
}

/*
 * In Numerical Recipes in C: The Art of Scientific Computing 
 * (William H. Press, Brian P. Flannery, Saul A. Teukolsky, William T. Vetterling;
 *  New York: Cambridge University Press, 1992 (2nd ed., p. 277))
 */
inline static long simRandomRange(long low, long high) {
    return low + (long) ( ((double) high)* (simRandom() / (SIM_RAND_MAX + 1.0)));
}


// #if defined (__MIC__)
//         _mm_delay_32(nticks);
// #else
//         ticks_wait(nticks);
// #endif


using namespace ff;


static inline void compute(long id, int nticks) {
    if (next == 0UL) {
        simSRandom(id + 1L);
    }
    long val = simRandomRange(1,nticks);
    ticks_wait(val);
    //for(volatile long k=0;k<val;++k) ;
}

int main(int argc, char *argv[]) {
    long numtasks = 1000000;
    int  nworkers = 3;
    int  nticks   = 1000;
    int  chunk    = -1;
    
    if (argc>1) {
        if (argc<4) {
            printf("use: %s numtasks nworkers ticks [chunk=(numtasks/nworkers)]\n", argv[0]);
            return -1;
        }
        numtasks = atol(argv[1]);
        nworkers = atoi(argv[2]);
        nticks   = atoi(argv[3]);
        if (argc == 5) 
            chunk = atoi(argv[4]);
    }
#if defined(USE_OPENMP)
    ffTime(START_TIME);
#pragma omp parallel for schedule(runtime) num_threads(nworkers)
    for(long j=0;j<numtasks;++j) {
        compute(omp_get_thread_num(),nticks);
    }
    ffTime(STOP_TIME);
    printf("%d Time  = %g (ms)\n", nworkers, ffTime(GET_TIME));
#elif defined(USE_TBB)
    tbb::task_scheduler_init init(nworkers);
    tbb::affinity_partitioner ap;
    
    ffTime(START_TIME);
    tbb::parallel_for(tbb::blocked_range<long>(0, numtasks, chunk),
                      [&] (const tbb::blocked_range<long>& r) {
                          for (long j=r.begin();j!=r.end();++j) {
                              tbb::tbb_thread::id tid = tbb::this_tbb_thread::get_id();
                              compute(*(long*)&tid,nticks);
                          }
                      }, ap);
    ffTime(STOP_TIME);
    printf("%d Time  = %g (ms)\n", nworkers, ffTime(GET_TIME));

#else

    FF_PARFOR_INIT(pf, nworkers);

    ffTime(START_TIME);
    FF_PARFOR_START(pf, j,0,numtasks,1, chunk, nworkers) {
        compute(_ff_thread_id,nticks);
    } FF_PARFOR_STOP(pf);
    ffTime(STOP_TIME);
    printf("%d Time  = %g (ms)\n", nworkers, ffTime(GET_TIME));

    FF_PARFOR_DONE(pf);
#endif
    return 0;
}
