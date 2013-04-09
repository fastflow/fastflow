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

static inline void F(double& M) {
    M = M* (sin(M)/cos(M));
}

MAPDEF1(map, F, double)

int main(int argc, char * argv[]) {
    if (argc<3) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " arraysize nworkers\n";
        return -1;
    }
    size_t size     =atoi(argv[1]);
    int    nworkers =atoi(argv[2]);

    /* init */
    double *M = new double[size];
    for(size_t j=0;j<size;++j) M[j]=((j+1)/(double)size);

#if defined(USE_OPENMP)
#pragma omp parallel for
    for(size_t i=0;i<size;++i) 
        M[i] = M[i]* (sin(M[i])/cos(M[i])); // F(&M[i],1);
#else
    MAP(map,double,M,size,nworkers);
    RUNMAP(map);
#endif
    printf("DONE\n");
    return 0;
}
