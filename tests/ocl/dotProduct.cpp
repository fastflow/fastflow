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
 * Author: Massimo Torquati <torquati@di.unipi.it> 
 * Date:   October 2014
 */

#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif

#define CHECK 1

#ifdef CHECK
#include "ctest.h"
#else
#define NACC 1
#endif
//#define FF_OPENCL_LOG

#include <ff/stencilReduceOCL.hpp>

using namespace ff;
#define BUILD_WITH_SOURCE 1
#if !defined(BUILD_WITH_SOURCE)

FF_OCL_MAP_ELEMFUNC_1D_ENV(mapf, float, a, float, b,
                     return (a * b);
);

FF_OCL_REDUCE_COMBINATOR(reducef, float, x, y,
        return (x+y);
);
#endif

struct oclTask: public baseOCLTask<oclTask, float, float> {
    oclTask():M(NULL),M2(NULL), Mout(NULL),result(0.0), size(0) {}
    oclTask(float *M, float *M2, size_t size):M(M),M2(M2),Mout(NULL),result(0.0),size(size) {
        Mout = new float[size];
        assert(Mout);
    }
    ~oclTask() { if (Mout) delete [] Mout; }
    void setTask(oclTask *t) { 
       setInPtr(t->M, t->size);
       setEnvPtr(t->M2, t->size);
       setOutPtr(t->Mout, t->size);
       setReduceVar(&(t->result));
     }
    float combinator(float const &x, float const &y) {return x+y;}

    float *M, *M2;
    float  *Mout, result;
    const size_t  size;
};

int main(int argc, char * argv[]) {
    size_t size = 1024;
    if (argc>1) size     =atol(argv[1]);
    printf("arraysize = %ld\n", size);
    
    float *M = new float[size];
    float *M2 = new float[size];

    for(size_t i=0;i<size;++i) {
        M[i] = 2.0f;
        M2[i] = 5.0f;
    }    

#if defined(CHECK)
    float r = 0.0;
    for(size_t j=0;j<size;++j) {
        r += M[j] * M2[j];
    }
#endif
    oclTask oclt(M, M2, size);
#if defined(BUILD_WITH_SOURCE)
    ff_mapReduceOCL_1D<oclTask> oclMR(oclt, "cl_code/dotProduct.cl", "mapf", "reducef", 0, nullptr, NACC);
#else
    ff_mapReduceOCL_1D<oclTask> oclMR(oclt, mapf, reducef, 0, nullptr, NACC);
#endif
    oclMR.saveBinaryFile(); oclMR.reuseBinaryFile();
    SET_DEVICE_TYPE(oclMR);

    ffTime(START_TIME);
    if (oclMR.run_and_wait_end()<0) {
        error("running map-reduce\n");
        return -1;
    }
    ffTime(STOP_TIME);

    printf("Time = %.2f (ms)\n", ffTime(GET_TIME));

    delete [] M;
    printf("res=%.2f\n", oclt.result);
#if defined(CHECK)
    if (r != oclt.result) {
    	printf("Wrong result, should be %.2f\n", r);
    	exit(1); //ctest
    }
    else printf("OK\n");
#endif
    return 0;
}
