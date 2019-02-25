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
 * Date  : February 2014
 *
 */
#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>

using namespace ff;

struct ReductionVars {
    ReductionVars():sum(0.0),diff(0.0) {}
    ReductionVars(double s, double d):sum(s),diff(d) {}

    ReductionVars& operator+=(const ReductionVars& v) {
        sum  += v.sum;
        diff += v.diff;  // NOTE is += and not -= because v.diff can be < 0
        return *this;
    }
    
    double sum;
    double diff;
};


int main(int argc, char *argv[]) {
    int arraySize= 10000000;
    int nworkers = 3;
    if (argc>1) {
        if (argc<3) {
            printf("use: %s arraysize nworkers\n", argv[0]);
            return -1;
        }
        arraySize= atoi(argv[1]);
        nworkers = atoi(argv[2]);
    }

    if (nworkers<=0) {
        printf("Wrong parameters values\n");
        return -1;
    }
    
    // creates the array
    double *A = new double[arraySize];
    double *B = new double[arraySize];

    ReductionVars R(1000.0, -1000.0), Rzero;

    ParallelForReduce<ReductionVars> pfr(nworkers);

    // init data
    for(int j=0; j<arraySize; ++j) {
        A[j]=j*3.14; B[j]=2.1*j;
    }
  
    auto reduceF = [](ReductionVars &R, const ReductionVars &r) { 
        R += r;
    };
  
    pfr.parallel_reduce(R, Rzero, 0, arraySize, 1, 0, [&](const long i, ReductionVars &R) {
            auto tmp = A[i]*B[i];
            R.sum  += tmp;
            R.diff -= tmp;
        }, reduceF, nworkers);
    
    printf("R sum=%g diff=%g\n", R.sum, R.diff);
    
    return 0;
}
