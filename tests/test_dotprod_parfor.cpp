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
 * The program computes the dot-product of 2 arrays.
 *  compile with -DUSE_OPENMP for the OpenMP version
 *  compile with -DUSE_TBB for the TBB version
 *
 */

#include <ff/parallel_for.hpp>

#if defined(USE_TBB) 
#include <numeric>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#endif

using namespace ff;

int main(int argc, char * argv[]) {    
    if (argc<3) {
        printf("use: %s arraysize nworkers [ntimes] [CHUNKSIZE]\n", argv[0]);
        return -1;
    }
    int arraySize= atoi(argv[1]);
    int nworkers = atoi(argv[2]);
    int NTIMES   = 1;
    int CHUNKSIZE= std::min(10000, arraySize/nworkers);

    if (argc>=4) NTIMES = atoi(argv[3]);
    if (argc==5) CHUNKSIZE = atoi(argv[4]);

    if (nworkers<=0) {
        printf("Wrong parameters values\n");
        return -1;
    }
    
    //printf("CHUNKSIZE=%d\n", CHUNKSIZE);

    // creates the array
    double *A = new double[arraySize];
    double *B = new double[arraySize];

    double sum = 5.0;
#if defined(USE_OPENMP)

    // init data
    #pragma omp parallel for schedule(static,(arraySize/nworkers))
    for(long j=0;j<arraySize;++j) {
        A[j]=j*3.14; B[j]=2.1*j;
    }

    ff::ffTime(ff::START_TIME);
    for(int z=0;z<NTIMES;++z) {

        // do work
#pragma omp parallel for default(shared)                                \
    schedule(runtime)                                                   \
    reduction(+:sum)                                                    \
    num_threads(nworkers)
        for(long i=0;i<arraySize;++i)
            sum += A[i]*B[i];
        
    } // for z

    ffTime(STOP_TIME);
    printf("omp %d Time = %g ntimes=%d\n", nworkers, ffTime(GET_TIME), NTIMES);

#elif defined(USE_TBB)


    tbb::task_scheduler_init init(nworkers);
    tbb::affinity_partitioner ap;

    // init data
    tbb::parallel_for(tbb::blocked_range<long>(0, arraySize, (arraySize/nworkers)),
                      [&] (tbb::blocked_range<long> &r) {
                          for(long j=r.begin(); j!=r.end(); ++j) {
                              A[j]=j*3.14; B[j]=2.1*j;
                          }
                      },ap);

    ff::ffTime(ff::START_TIME);
    for(int z=0;z<NTIMES;++z) {
        // do work
        sum += tbb::parallel_reduce(tbb::blocked_range<long>(0, arraySize, CHUNKSIZE), double(0),
                                    [=] (tbb::blocked_range<long> &r, double in) {
                                        return std::inner_product(A+r.begin(),
                                                                  A+r.end(),
                                                                  B+r.begin(),
                                                                  in,
                                                                  std::plus<double>(),
                                                                  std::multiplies<double>());                                                              
                                    }, std::plus<double>(), ap);
        
    }
    ffTime(STOP_TIME);
    printf("tbb %d Time = %g ntimes=%d\n", nworkers, ffTime(GET_TIME), NTIMES);

#else // (default) FastFlow version
    
    FF_PARFORREDUCE_INIT(dp, double, nworkers);

    // init data
    FF_PARFOR_BEGIN(init, j,0,arraySize,1, (arraySize/nworkers),nworkers) {
        A[j]=j*3.14; B[j]=2.1*j;
    } FF_PARFOR_END(init);

    //auto Fsum = [](double& v, double elem) { v += elem; };
        
    ff::ffTime(ff::START_TIME);    
    for(int z=0;z<NTIMES;++z) {
        // do work
        FF_PARFORREDUCE_START(dp, sum, 0.0, i,0,arraySize,1, CHUNKSIZE, nworkers) { 
            sum += A[i]*B[i];
        //} FF_PARFORREDUCE_F_STOP(dp, sum, Fsum);    // this is just a different form
        } FF_PARFORREDUCE_STOP(dp, sum, +);    
    }
    ffTime(STOP_TIME);
    printf("ff %d Time = %g ntimes=%d\n", nworkers, ffTime(GET_TIME), NTIMES);
    FF_PARFORREDUCE_DONE(dp);
#endif
    
    printf("Sum =%g\n", sum);
    return 0;
}
