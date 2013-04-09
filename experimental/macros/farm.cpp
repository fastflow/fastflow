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

static inline void F(double taskin, double*& taskout) {
    taskout = new double(taskin * (sin(taskin)/cos(taskin)));
}

FFnode(F,double,double);

int main(int argc, char * argv[]) {
    if (argc<3) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " arraysize nworkers\n";
        return -1;
    }
    size_t numTasks =atoi(argv[1]);
    int    nworkers =atoi(argv[2]);

    double* res = NULL;
    FARMA(farmA, F, nworkers);
    RUNFARMA(farmA);
    for(size_t i=0;i<numTasks;++i) {
        FARMAOFFLOAD(farmA, new double(i*3.14));
        if (FARMAGETRESULTNB(farmA, &res)) {
            printf("got %g\n", *res);
            delete res;
        }
    }
    FARMAEND(farmA);
    while(FARMAGETRESULT(farmA, &res)) {
        printf("got %g\n", *res);
        delete res;
    }

    FARMAWAIT(farmA);
        
    printf("DONE\n");
    return 0;
}
