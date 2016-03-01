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
 */
// This is the same of the oclMap test but the source code is loaded from a file.
// The compiled version of the OpenCL program is saved in binary file (.bin) 
// so that at the next run the program is not recompiled.


#if !defined(FF_OPENCL)
// needed to enable the OpenCL FastFlow run-time
#define FF_OPENCL
#endif

#include <ff/stencilReduceOCL.hpp>
using namespace ff;

#define CHECK 1
#ifdef CHECK
#include "ctest.h"
#else
#define NACC 1
#endif


struct oclTask: public baseOCLTask<oclTask, float> {
    oclTask():M(NULL),size(0) {}
    oclTask(float *M, size_t size):M(M),size(size) {}
    void setTask(oclTask *t) { 
        setInPtr(t->M, t->size);
        setOutPtr(t->M, t->size);
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
    ff_mapOCL_1D<oclTask> oclMap1(oclt, "cl_code/oclMap.cl","mapf1",nullptr,NACC);
    SET_DEVICE_TYPE(oclMap1);

    oclMap1.saveBinaryFile();  // save the compiled version in cl_code/oclMap.cl.bin
    oclMap1.reuseBinaryFile(); // if the binary file is present it will be used
    if (oclMap1.run_and_wait_end()<0) {
        error("running oclMap1\n");
        return -1;
    }
    
    ff_mapOCL_1D<oclTask> oclMap2(oclt, "cl_code/oclMap.cl","mapf2",nullptr,NACC);
    SET_DEVICE_TYPE(oclMap2);
    oclMap2.reuseBinaryFile(); 
    if (oclMap2.run_and_wait_end()<0) {
        error("running oclMap2\n");
        return -1;
    }


#if defined(CHECK)
	for (size_t i = 0; i < size; ++i) {
		if ((M_[i]+1) != M[i]) {
            printf("Error\n");
            exit(1);
        }
	}
    printf("Result correct\n");
#endif
    return 0;
}
