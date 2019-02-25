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
 * Date  : September 2014
 *
 */
/* Simple example demonstrating the usege of a single ParallelForReduce pattern 
 * for the reduction of multiple variables having different type.
 *
 */

#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>

using namespace ff;

struct ReductionVars {
    ReductionVars():u(0L) {}

    const double& sumd() const { return u.sumd; }
    double& sumd()             { return u.sumd;}
    const long& suml() const   { return u.suml; }
    long& suml()               { return u.suml;}

    ReductionVars& operator+=(const double& v) {
        u.sumd  += v;
        return *this;
    }
    ReductionVars& operator+=(const long& v) {
        u.suml  += v;
        return *this;
    }

    union u {
        u(long x):sumd(x) {}
        double sumd;
        long   suml;
    } u;
};


int main(int argc, char *argv[]) {
    int arraySize= 10000000;
    int nworkers = 2;
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

    ReductionVars R, Rzero;
    ParallelForReduce<ReductionVars> pfr(nworkers);

    // init data
    for(int j=0; j<arraySize; ++j) A[j]=j*3.14;

    {
        auto reduceF = [](ReductionVars &R, const ReductionVars &r) { 
            R += r.sumd();
        };  
        pfr.parallel_reduce(R, Rzero, 0, arraySize, 1, 0, [&](const long i, ReductionVars &R) {
                R  += A[i];
            }, reduceF, nworkers);
        
        printf("R (double)\tsum=%g \n", R.sumd());
    }
    {
        R = Rzero;
        auto reduceF = [](ReductionVars &R, const ReductionVars &r) { 
            R += r.suml();
        };  
        pfr.parallel_reduce(R, Rzero, 0, arraySize, 1, 0, [&](const long i, ReductionVars &R) {
                R  += (long)A[i];
            }, reduceF, nworkers);
        
        printf("R (long)\tsum=%ld \n", R.suml());
    }
    return 0;
}
