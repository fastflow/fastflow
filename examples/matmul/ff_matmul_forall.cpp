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
        std::cerr << "use: " << argv[0] << " nworkers size [chunk=0 K=5 check=0]\n";
        return -1;
    }
    int    chunk    = 0; // default static scheduling
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
   
#if 0
    double *A = (double*)_mm_malloc(N*N*sizeof(double),64);
    double *B = (double*)_mm_malloc(N*N*sizeof(double),64);
    double *C = (double*)_mm_malloc(N*N*sizeof(double),64);
#endif
    double *A = (double*)malloc(N*N*sizeof(double));
    double *B = (double*)malloc(N*N*sizeof(double));
    double *C = (double*)malloc(N*N*sizeof(double));


    assert(A && B && C);

#if defined(__MIC__)
    const char worker_mapping[]="1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 0, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 130, 134, 138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186, 190, 194, 198, 202, 206, 210, 214, 218, 222, 226, 230, 234, 237, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 238, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 239";
    threadMapper::instance()->setMappingList(worker_mapping);
#endif

    ParallelFor pf(nworkers);

    // init data
    pf.parallel_for(0,N,1,chunk, [&](const long i) {
            for(long j=0;j<N;++j) {
                A[i*N+j] = (i+j)/(double)N;
                B[i*N+j] = i*j*3.14;
                C[i*N+j] = 0;
            }
        },nworkers);
    
    auto lambda = [&A,&B,&C,N](const long i) {
        //helping compiler vectorization
#if defined(__ITEL_COMPILER)
        double *restrct _A = A, *restric _B=B, *restrict _C=C;
#else
        double *__restrict__ _A = A, *__restrict__ _B=B, *__restrict__ _C=C;
#endif
        const long _N = N;
        const long iN = i*_N;
        
        for(long j=0;j<_N;++j) {
            for(long k=0;k<_N;++k)
                _C[iN+j] += _A[iN+k]*_B[k*_N+j];
        }
    };

    ffTime(START_TIME);
    for(int q=0;q<K;++q) {
        pf.parallel_for(0,N,1,chunk,lambda,nworkers);
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
