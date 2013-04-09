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
 *
 */
#include <vector>
#include <iostream>
#include <cmath>

#include <ffmacros.h>

const int N=2048;
static double A[N][N];
static double B[N][N];
static double C[N][N];

static inline void F(long i, void*& taskout) {
    double long _C=0;
    for(long j=0;j<N;++j) {
        for(long k=0;k<N;++k)
            _C += A[i][k]*B[k][j];        
        C[i][j] = _C; _C=0;
    }
    taskout=GO_ON;
}
FFnode(F,long,void);

int main(int argc, char * argv[]) {
    if (argc<2) {
        std::cerr << "use: " << argv[0] << " nworkers\n";
        return -1;
    }
    int    nworkers =atoi(argv[1]);

    for(long i=0;i<N;++i) 
        for(long j=0;j<N;++j) {
            A[i][j] = (i+j)/N;
            B[i][j] = i*j*3.14;
            C[i][j] = 0;
        }
  
    ffTime(START_TIME);
#if defined(USE_OPENMP)
    ++nworkers; // to make the compiler appy!
#pragma omp parallel for
  for(int i=0;i<N;++i) 
	for(int j=0;j<N;++j)
	  for(int k=0;k<N;++k)
		C[i][j] += A[i][k]*B[k][j];
#else
    FARMA2(farmA, F, nworkers);
    RUNFARMA(farmA);
    for(long i=0;i<N;++i)
        FARMAOFFLOAD(farmA, new long(i));
    FARMAEND(farmA);
    FARMAWAIT(farmA);
#endif
    printf("Time = %g (ms)\n", ffTime(STOP_TIME));
    return 0;
}
