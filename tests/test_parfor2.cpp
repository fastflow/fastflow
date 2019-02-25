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

#include <cstdlib>
#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>

using namespace ff;

int main(int argc, char *argv[]) {
    long size     = 100000;
    long chunk    = 111;
    int  nworkers = 3;
    int  ntimes   = 3;

    if (argc>1) {
        if (argc<5) {
            printf("use: %s size nworkers ntimes chunk\n", argv[0]);
            return -1;
        }
        size     = atol(argv[1]);
        nworkers = atoi(argv[2]);
        ntimes   = atoi(argv[3]);
        chunk    = atol(argv[4]);
    }
    long *A = new long[size];

    ParallelForReduce<long> pfr(nworkers, true);
    long sum;
    for(int k0=0;k0<3;++k0) {  // this loop just tests the threadPause()
    
        sum = 0;
        for(int k=0;k<ntimes; ++k) {
            auto loop1 = [&A,k](const long j) { A[j]=j+k; };
            auto loop2 = [&A](const long i, long& sum) { sum += A[i];};
            auto Fsum = [](long& v, const long elem) { v += elem; };
            
            pfr.parallel_for(0L,size,1L,chunk,loop1,(std::min)(k+1, nworkers));
            printf("loop1 using %d workers\n", (std::min)(k+1, nworkers));
            pfr.parallel_reduce(sum,0L,0L,size,1L,chunk,loop2,Fsum,nworkers);
            printf("loop2 using %d workers\n", nworkers);
        }
        pfr.threadPause();

        printf("sum = %ld\n", sum);
    } // k0

    printf("loop done\n");
    printf("sum = %ld\n", sum);

    return 0;
}
