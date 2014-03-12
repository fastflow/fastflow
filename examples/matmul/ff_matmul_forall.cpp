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
 * NxN square matrix multiplication: C = A x B
 * 
 *
 * Author: Massimo Torquati <torquati@di.unipi.it> <massimotor@gmail.com>
 */
#include <vector>
#include <iostream>
#include <cmath>
#include <ff/parallel_for.hpp>

using namespace ff;

int main(int argc, char * argv[]) {
    if (argc<3) {
        std::cerr << "use: " << argv[0] << " nworkers size [chunk=-1 K=5 check=0]\n";
        return -1;
    }
    int    chunk    = -1;
    bool   check    = false;  
    int    K        = 5;
    int    nworkers =atoi(argv[1]);
    long   N        =atol(argv[2]);
    assert(N>0);
    if (argc>=4)  chunk    =atoi(argv[3]);
    if (argc>=5)  K        =atoi(argv[4]);
    if (argc==6) {
        check=true;  // checks result
        if (K>1) printf("K is reset to 1 because of checking results\n");
        K=1;         // reset K
    }
   
    double *A = (double*)malloc(N*N*sizeof(double));
    double *B = (double*)malloc(N*N*sizeof(double));
    double *C = (double*)malloc(N*N*sizeof(double));
    assert(A && B && C);

    ParallelFor pf(nworkers);

    // init data
    pf.parallel_for(0,N,1,chunk, [&](const long i) {
            for(long j=0;j<N;++j) {
                A[i*N+j] = (i+j)/(double)N;
                B[i*N+j] = i*j*3.14;
                C[i*N+j] = 0;
            }
        },nworkers);
    
    ffTime(START_TIME);
    for(int q=0;q<K;++q) {
        pf.parallel_for(0,N,1,chunk,[&A,&B,&C,N](long i) {
                //helping compiler vectorization
#if defined(__ITEL_COMPILER)
                double *restrct _A = A, *restric _B=B, *restrict _C=C;
#else
                double *__restrict__ _A = A, *__restrict__ _B=B, *__restrict__ _C=C;
#endif
                for(long j=0;j<N;++j) {
                    PRAGMA_IVDEP;
                    for(long k=0;k<N;++k)
                        _C[i*N+j] += _A[i*N+k]*_B[k*N+j];
                }
            },nworkers);
    }
    ffTime(STOP_TIME);
    printf("%d Time = %g (ms) K=%d\n", nworkers, ffTime(GET_TIME)/(1.0*K), K);

    // checking the result
    if (check) {
        double R=0;
        for(long i=0;i<N;++i)
            for(long j=0;j<N;++j) {
                for(long k=0;k<N;++k)
                    R += A[i*N+k]*B[k*N+j];
                
                if (abs(C[i*N+j]-R)>1e-06) {
                    std::cerr << "Wrong result\n";                    
                    return -1;
                }
                R=0;
            }
        std::cout << "OK\n";
    }
    return 0;
}
