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
//#include <mm_malloc.h>

#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>

#if defined(USE_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#endif


using namespace ff;

int main(int argc, char *argv[]) {
    int  nworkers = 3;
    long numtasks = 10000000*nworkers;
    int   chunk = 100;
    if (argc>1) {
        if (argc<3) {
            printf("use: %s numtasks nworkers [chunk=(numtasks/nworkers)]\n", argv[0]);
            return -1;
        }
        numtasks = atol(argv[1]);
        nworkers = atoi(argv[2]);
        if (argc == 4) 
            chunk = atoi(argv[3]);
    }
    
    long *V;
    if (posix_memalign((void**)&V, 64, numtasks*sizeof(long)) != 0) abort();
    //long *V = new long[numtasks];
    
#if defined(USE_OPENMP)
    ffTime(START_TIME);
#pragma omp parallel for schedule(runtime) num_threads(nworkers)
    for(long j=0;j<numtasks;++j) {
        V[j]=j;
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
                              V[j] = j;
                          }
                      }, ap);
    ffTime(STOP_TIME);
    printf("%d Time  = %g (ms)\n", nworkers, ffTime(GET_TIME));
#else

#if 0
    // macroes interface
    FF_PARFOR_INIT(pf, nworkers);
    ffTime(START_TIME);
    FF_PARFOR_START(pf, j,0,numtasks,1, chunk, nworkers) {
        V[j]=j;
    } FF_PARFOR_STOP(pf);
    ffTime(STOP_TIME);
    printf("%d Time  = %g (ms)\n", nworkers, ffTime(GET_TIME));
    FF_PARFOR_DONE(pf);
#else
    ParallelFor pf(nworkers);
    ffTime(START_TIME);
    pf.parallel_for(0,numtasks,1,chunk, [&V](const long j) {
            V[j]=j;
        });
    ffTime(STOP_TIME);
    printf("%d Time  = %g (ms)\n", nworkers, ffTime(GET_TIME));
#endif
#endif
    return 0;
}
