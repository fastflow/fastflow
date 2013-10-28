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
 * This tests shows how to use FF_PARFOR_INIT/DONE together with 
 * FF_PARFOR_START/STOP. 
 *
 */

#include <cstdlib>
#include <ff/parallel_for.hpp>

using namespace ff;

int main(int argc, char *argv[]) {
    if (argc<5) {
        printf("use: %s size nworkers ntimes chunk\n", argv[0]);
        return -1;
    }
    const long size     = atol(argv[1]);
    const int  nworkers = atoi(argv[2]);
    const int  ntimes   = atoi(argv[3]);
    const int  chunk    = atoi(argv[4]);
    long *A = new long[size];

    FF_PARFOR_INIT(pf1, nworkers);
    FF_PARFORREDUCE_INIT(pf2, long, nworkers);

    long sum=0.0;
    for(int k=0;k<ntimes; ++k) {

        for(int j = 0; j< size; ++j) 
            A[j] = j+k;

        FF_PARFOR_START(pf1, j,0,size,1, 1, std::min(k+1, nworkers)) {
            A[j]=j+k;
        } FF_PARFOR_STOP(pf1);
        printf("pf1 done using %d workers\n", std::min(k+1,nworkers));


        FF_PARFORREDUCE_BEGIN(pf2, sum, 0, i,0,size,1, chunk, std::min(k+2,nworkers)) { 
            sum += A[i];
        } FF_PARFORREDUCE_END(pf2, sum, +);    
        printf("pf2 done using %d workers\n", std::min(k+2,nworkers));
        
    } // k
    
    printf("loop done\n");

    FF_PARFOR_DONE(pf1);
    FF_PARFORREDUCE_DONE(pf2);

    printf("sum = %ld\n", sum);
    return 0;
}
