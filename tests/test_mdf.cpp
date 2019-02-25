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
 * simple test for the MDF pattern.
 *                                               ________________
 *                                             /        \        \
 *                                             |         |        |
 *    A = A + B;        // sum2          A     B     C   |     D  |
 *    C = C + B;        // sum2          |     |     |   |     |  |
 *    D = D + B;        // sum2           \   / \   /     \   /   |
 *    D = A + C + D;    // sum3             +     +         +     |
 *    A = A + D;        // sum2          -- A     C ---     D     |
 *    B = B + D;        // sum2         |   |     |    |    |     |
 *    C = C + D;        // sum2         |    \     \    -- / ---- |--------
 *    res =(+)(A,B,C,D) // reduce       |     -----  + ---        |        |
 *                                      |            D ---------- | - ---- | --
 *                                      |            |            |  \     |   |
 *                                       \          / \           /   \   /    |
 *                                        ---- + ---   ---- + ----      +      |
 *                                            A            B            C      |
 *                                            |            |            |      |
 *                                            |            |            |      |
 *                                             -------   reduce(+) ----- ------
 *                                                         |
 *                                                        res
 */

#include <ff/ff.hpp>
#include <ff/mdf.hpp>
#include <ff/parallel_for.hpp>

using namespace ff;

const bool check = true;
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
// res = reduce(+)(X, Y, Z, K)
void reduce(long *res, long *X, long *Y, long *Z, long *K, const long size) {
    ParallelForReduce<long> pfr;
    auto Freduce = [](long &res, const long elem) { res += elem;};
    pfr.parallel_reduce(*res,0, 
			0,size, [&X,&Y,&Z,&K](const long i, long &res) {
			    res += X[i] + Y[i] + Z[i] + K[i];
			}, Freduce, 2);    
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
    {
	const param_info _1={(uintptr_t)A,ff::INPUT};
	const param_info _2={(uintptr_t)B,ff::INPUT};
	const param_info _3={(uintptr_t)A,ff::OUTPUT};
	Param.push_back(_1); Param.push_back(_2); Param.push_back(_3);
	mdf->AddTask(Param, sum2, A,B,MYSIZE);
    }

    // C = C + B;
    {
	Param.clear();
	const param_info _1={(uintptr_t)C,ff::INPUT};
	const param_info _2={(uintptr_t)B,ff::INPUT};
	const param_info _3={(uintptr_t)C,ff::OUTPUT};
	Param.push_back(_1); Param.push_back(_2); Param.push_back(_3);
	mdf->AddTask(Param, sum2, C,B,MYSIZE);
    }

    // D = D + B;
    {
	Param.clear();
	const param_info _1={(uintptr_t)D,ff::INPUT};
	const param_info _2 = { (uintptr_t)B, ff::INPUT };
	const param_info _3={(uintptr_t)D,OUTPUT};
	Param.push_back(_1); Param.push_back(_2); Param.push_back(_3);
	mdf->AddTask(Param, sum2, D,B,MYSIZE);
    }

    // D = A + C + D;
    { 
	Param.clear();
	const param_info _1={(uintptr_t)A,ff::INPUT};
	const param_info _2 = { (uintptr_t)C, ff::INPUT };
	const param_info _3 = { (uintptr_t)D, ff::INPUT };
	const param_info _4={(uintptr_t)D,ff::OUTPUT};
	Param.push_back(_1); Param.push_back(_2); Param.push_back(_3); Param.push_back(_4);
	mdf->AddTask(Param, sum3, D,A,C,MYSIZE);
    }

    // A = A + D;
    {
	Param.clear();
	const param_info _1 = { (uintptr_t)A, ff::INPUT };
	const param_info _2 = { (uintptr_t)D, ff::INPUT };
	const param_info _3={(uintptr_t)A,ff::INPUT};
	Param.push_back(_1); Param.push_back(_2); Param.push_back(_3);
	mdf->AddTask(Param, sum2, A, D,MYSIZE);
    }

    // B = B + D;
    {
	Param.clear();
	const param_info _1 = { (uintptr_t)B, ff::INPUT };
	const param_info _2 = { (uintptr_t)D, ff::INPUT };
	const param_info _3={(uintptr_t)B,ff::OUTPUT};
	Param.push_back(_1); Param.push_back(_2); Param.push_back(_3);
	mdf->AddTask(Param, sum2, B, D,MYSIZE);
    }

    // C = C + D;
    {
	Param.clear();
	const param_info _1 = { (uintptr_t)C, ff::INPUT };
	const param_info _2 = { (uintptr_t)D, ff::INPUT };
	const param_info _3={(uintptr_t)C, ff::OUTPUT };
	Param.push_back(_1); Param.push_back(_2); Param.push_back(_3);
	mdf->AddTask(Param, sum2, C, D,MYSIZE);
    }

    // res = reduce(+)(A,B,C,D)
    {
	Param.clear();
	const param_info _1 = { (uintptr_t)A, ff::INPUT };
	const param_info _2 = { (uintptr_t)B, ff::INPUT };
	const param_info _3 = { (uintptr_t)C, ff::INPUT };
	const param_info _4 = { (uintptr_t)D, ff::INPUT };
	const param_info _5={(uintptr_t)&P->res,ff::OUTPUT};
	Param.push_back(_1); Param.push_back(_2); Param.push_back(_3); Param.push_back(_4); Param.push_back(_5);
	mdf->AddTask(Param, reduce, &P->res, A,B,C,D,MYSIZE);
    }
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
        if (P.res != res) {
            printf("WRONG RESULT\n");
            return -1;
        }
    }
    return 0;

}
