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
 *         
 */
/*
 *    A = A + B;    
 *    C = C + B;    
 *    D = D + A + C;    
 */

#include <ff/mdf.hpp>
#include <ff/parallel_for.hpp>

using namespace ff;

const bool check = false;    // true to check the result
const long MYSIZE = (1 << 20);

// X = X + Y
void sum2(long *X, long *Y, const long size) {
    for(long i=0;i<size;++i)
	X[i] += Y[i];
}
// X = X + Y + Z
void sum3(long *X, long *Y, long *Z, const long size) {
    for(long i=0;i<size;++i)
	X[i] += Y[i] + Z[i];
}


template<typename T>
struct Parameters {
    long *A,*B,*C,*D;
    long res;
    T* mdf;
};


void taskGen(Parameters<ff_mdf > *const P){
    long *A = P->A;
    long *B = P->B;
    long *C = P->C;
    long *D = P->D;

    std::vector<param_info> Param;
    auto mdf = P->mdf;

    // A = A + B;
    mdf::task_t *task1 = mdf->AddTask(sum2, A,B,MYSIZE);

    // C = C + B;
    mdf::task_t *task2 = mdf->AddTask(sum2, C,B,MYSIZE);

    // D = D + B;
    const param_info _1={(uintptr_t)task1, ff::INPUT};
    const param_info _2={(uintptr_t)task2, ff::INPUT};
    Param.push_back(_1); Param.push_back(_2);
    mdf->AddTask(Param, sum3, D,A,C,MYSIZE);

}


int main() {
    long *A = new long[MYSIZE];
    long *B = new long[MYSIZE];
    long *C = new long[MYSIZE];
    long *D = new long[MYSIZE];

    assert(A && B && C && D);

    for(long i=0;i<MYSIZE;++i) {
        A[i] = 0; B[i] = i;
        C[i] = 1; D[i] = i+1;
    }
    
    Parameters<ff_mdf > P;
    ff_mdf dag(taskGen, &P, 16, 3);
    P.A=A,P.B=B,P.C=C,P.D=D,P.mdf=&dag;P.res=0;
    
    dag.run_and_wait_end();

    // printing result
    printf("result = %ld\n", P.res);
    
    if (check) {
        // re-init data
        for(long i=0;i<MYSIZE;++i) {
            A[i] = 0; B[i] = i;
            C[i] = 1; D[i] = i+1;
        }
        long res=0;
        
		sum2(A, B, MYSIZE);
		sum2(C, B, MYSIZE);
		sum2(D, B, MYSIZE);
		sum3(D, A, C, MYSIZE);
		sum2(A, D, MYSIZE);
		sum2(B, D, MYSIZE);
		sum2(C, D, MYSIZE);
		reduce(&res, A, B, C, D, MYSIZE);
        printf("result = %ld\n",res);
    }
    return 0;

}
