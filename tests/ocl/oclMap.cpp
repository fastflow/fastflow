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
/* Author: Massimo Torquati
 *         torquati@di.unipi.it
 *
 *
 * Equivalent annotated code with REPARA attributes:
 *
 *  const size_t N = size;
 *  std::vector<float> M(N);
 *
 *  [[rpr::kernel, rpr::in(A), rpr::out(A), rpr::target(GPU)]]
 *  for (int i=0;<N;++i)
 *     A[i] = i;
 *
 *  [[rpr::kernel, rpr::in(A), rpr::out(A), rpr::target(GPU)]]
 *  for (int i=0;<N;++i)
 *     A[i] = A[i] + 1.0;
 *
 */

#if !defined(FF_OPENCL)
// needed to enable the OpenCL FastFlow run-time
#define FF_OPENCL
#endif

#include <ff/stencilReduceOCL.hpp>
using namespace ff;


// to check the result
#define CHECK 1
#ifdef CHECK
#include "ctest.h"
#else
#define NACC 1
#endif

// kernel 1
FF_OCL_MAP_ELEMFUNC(mapf1, float, elem, i,
                    return i;
                    );

// kernel 2
FF_OCL_MAP_ELEMFUNC(mapf2, float, elem, useless,
                    (void)useless;
                    return (elem+1.0);
                    );

// the OpenCL interface type
struct oclTask: public baseOCLTask<oclTask, float> {
    oclTask():M(NULL),size(0) {}
    oclTask(float *M, size_t size):M(M),size(size) {}
    void setTask(oclTask *t) { 
        setInPtr(t->M, t->size, MemoryFlags());  // set host input pointer
        setOutPtr(t->M, t->size, MemoryFlags());          // set host output pointer
    }

    float *M;
    const size_t  size;
};

int main(int argc, char * argv[]) {
    size_t size=2048;
    if(argc>1) size     =atol(argv[1]);
    printf("arraysize = %ld\n", size);

#ifdef CHECK
    std::vector<float> M_(size);
    for(size_t j=0;j<size;++j) M_[j]=j;
#endif

    std::vector<float> M(size);

    oclTask oclt(const_cast<float*>(M.data()), size);
    ff_mapOCL_1D<oclTask> oclMap1(oclt, mapf1, nullptr, NACC);
    // just selects whether the execution is on the CPU or GPU on the base of what
    // is written in the ctest.h file
    SET_DEVICE_TYPE(oclMap1); 
    if (oclMap1.run_and_wait_end()<0) {
        error("running oclMap1\n");
        return -1;
    }

    ff_mapOCL_1D<oclTask> oclMap2(oclt, mapf2, nullptr, NACC);
    // just selects whether the execution is on the CPU or GPU on the base of what
    // is written in the ctest.h file
    SET_DEVICE_TYPE(oclMap2); 
    if (oclMap2.run_and_wait_end()<0) {
        error("running oclMap2\n");
        return -1;
    }

#if defined(CHECK)
	for (size_t i = 0; i < size; ++i) {
		if ((M_[i]+1) != M[i]) {
            printf("Error\n");
            exit(1); //ctest
        }
	}
    printf("Result correct\n");
#endif
    return 0;
}
