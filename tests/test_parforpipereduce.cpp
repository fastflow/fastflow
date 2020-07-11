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
/* Author: Massimo
 * Date  : July 2014
 */

#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>


using namespace ff;

int main(int argc, char * argv[]) {    
    int arraySize= 10000;
    int nworkers = 3;
    int NTIMES   = 5;
    int CHUNKSIZE= (std::min)(100, arraySize/nworkers);
    if (argc>1) {
        if (argc<3) {
            printf("use: %s arraysize nworkers [ntimes] [CHUNKSIZE]\n", argv[0]);
            return -1;
        }
        arraySize= atoi(argv[1]);
        nworkers = atoi(argv[2]);
    
        if (argc>=4) NTIMES = atoi(argv[3]);
        if (argc==5) CHUNKSIZE = atoi(argv[4]);
    }

    if (nworkers<=0) {
        printf("Wrong parameters values\n");
        return -1;
    }
    

    // creates the array
    long *A = new long[arraySize];
    long *B = new long[arraySize];

    long *R = new long[arraySize];

#if 1
    {        
        parallel_for(0,arraySize,1,CHUNKSIZE, [&](const long j) { A[j]=j; B[j]=2*j+1;});
        ParallelForPipeReduce<std::vector<long>* > pfr(nworkers,true); // spinwait is set to true
        pfr.disableScheduler();
        
        auto Map = [&](const long start, const long stop, const int /*thid*/, ff_buffernode &node) {
            if (start == stop) return;
            std::vector<long>*  C = new std::vector<long>;
            C->reserve(stop-start);
            for(long i=start;i<stop;++i)  {
                // waste some time to simulate a more complex computation
                ticks_wait(1000); //for(volatile long m=0;m<50;++m); 
                C->push_back(A[i]*B[i]);
            }
            node.ff_send_out(C);
        };
        auto Reduce = [&](std::vector<long>* v) {
            const std::vector<long> &V = *v;
            for(size_t i=0;i<V.size();++i) {
                R[V[i] % arraySize] = V[i];
            }
            delete v;
        };

        ff::ffTime(ff::START_TIME);    
        for(int z=0;z<NTIMES;++z) {
            pfr.parallel_reduce_idx(0, arraySize,1,CHUNKSIZE, Map, Reduce);
        }
        ffTime(STOP_TIME);
        printf("ff %d Time = %g ntimes=%d\n", nworkers, ffTime(GET_TIME), NTIMES);
        printf("R[0]=%ld R[1]=%ld R[5]=%ld R[10]=%ld\n", R[0], R[1], R[5], R[10]);
    }
#else
    // (default) FastFlow version
    {
        parallel_for(0,arraySize,1,CHUNKSIZE, [&](const long j) { A[j]=j; B[j]=2*j+1;});
        ParallelForReduce<double> pfr(nworkers,true); // spinwait is set to true
        pfr.disableScheduler();

        std::vector<std::vector<long> > C(nworkers);;
        
        ff::ffTime(ff::START_TIME);    
        for(int z=0;z<NTIMES;++z) {
            pfr.parallel_for_thid(0, arraySize,1,CHUNKSIZE,
                                  [&](const long i, const int thid) { 
                                      // waste some time to simulate a more complex computation
                                      for(volatile long m=0;m<50;++m); 
                                      C[thid].push_back(A[i]*B[i]);
                                  }
                                  );
        }
        
        for(long k=0;k<nworkers;++k)
            for(size_t q=0;q<C[k].size();++q){
                R[C[k][q] % arraySize] = C[k][q];
            }

        ffTime(STOP_TIME);
        printf("ff %d Time = %g ntimes=%d\n", nworkers, ffTime(GET_TIME), NTIMES);
        printf("R[0]=%ld R[1]=%ld R[5]=%ld R[10]=%ld\n", R[0], R[1], R[5], R[10]);
    }
#endif

    return 0;
}
