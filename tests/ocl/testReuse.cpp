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
 *         (February 2016)       
 *  
 */

// to check the result
#define CHECK 1
#ifdef CHECK
#include "ctest.h"
#else
#define NACC 1
#endif

#include <cassert>
#include <ff/stencilReduceOCL.hpp>
using namespace ff;

struct Task1 {
    Task1(std::vector<float> &A):A(A) {}
    std::vector<float> &A;
};
struct Task2 {
    Task2(std::vector<float> &A, std::vector<float> &S):A(A),S(S) {}
    std::vector<float> &A;
    std::vector<float> &S;
};

/* ---------i---- OpenCL task types --------------------- */
struct oclTask1: public baseOCLTask<Task1, float> {
    oclTask1() {}
    void setTask(Task1 *t) { 
        assert(t);
        MemoryFlags mfin(CopyFlags::COPY, ReuseFlags::DONTREUSE, ReleaseFlags::DONTRELEASE);
        MemoryFlags mfenv(CopyFlags::DONTCOPY, ReuseFlags::DONTREUSE, ReleaseFlags::DONTRELEASE);
        MemoryFlags mfout(CopyFlags::DONTCOPY, ReuseFlags::DONTREUSE, ReleaseFlags::DONTRELEASE);
        
        setInPtr(t->A.data(), t->A.size(), mfin);  // copied
        setEnvPtr(t->A.data(), t->A.size(), mfenv); 
        setOutPtr(t->A.data(), t->A.size(), mfout); // not copied back 
    }
};

struct oclTask2: public baseOCLTask<Task2, float> {
    oclTask2() {}
    void setTask(Task2 *t) { 
        assert(t);
        MemoryFlags mfin(CopyFlags::DONTCOPY, ReuseFlags::REUSE, ReleaseFlags::DONTRELEASE);
        MemoryFlags mfout(CopyFlags::COPY, ReuseFlags::DONTREUSE, ReleaseFlags::DONTRELEASE);

        setInPtr(t->A.data(), t->A.size(),  mfin);  // not copied, reusing data
        setEnvPtr(t->S.data(), t->S.size(), mfout); // copied 
        setOutPtr(t->S.data(), t->S.size(), mfout); // copied back 
    }
};


// utility function
static void initInput(std::vector<float> &A, std::vector<float> &S) {
    for(size_t i=0; i<A.size(); ++i) {
        A[i] = i;
        S[i] = 0.0f;
    }
}

int main(int argc, char * argv[]) {
    const size_t max_size   = 10;

    std::vector<float> S(max_size);
    std::vector<float> A(max_size);
    
    initInput(A,S);

    // OpenCL device shared allocator
    ff_oclallocator alloc;

    // instances of the first and second task
    Task1 t1(A);    
    Task2 t2(A,S);

#if !defined(BINARY_OCL_CODE)
    //FF_OCL_MAP_ELEMFUNC_1D(mapf1, float, a, return a / (a*a + a + 1); );    
    FF_OCL_STENCIL_ELEMFUNC_ENV(mapf1, float, useless1, i, A, float, useless2, return A[i] / (A[i]*A[i] + A[i] + 1);  );
    ff_mapOCL_1D<Task1, oclTask1> oclMap1(t1, mapf1, &alloc); 
#else
    ff_mapOCL_1D<Task1, oclTask1> oclMap1(t1, "cl_code/reusek1.cl", &alloc); 
#endif
    SET_DEVICE_TYPE(oclMap1); 
    oclMap1.saveBinaryFile();  // save the compiled version in cl_code/oclMap.cl.bin
    oclMap1.reuseBinaryFile(); // if the binary file is present it is used

    if (oclMap1.run_and_wait_end()<0) {
        error("running oclMap1\n");
        return -1;
    }
    
#if !defined(BINARY_OCL_CODE)
    //FF_OCL_MAP_ELEMFUNC_1D_ENV(mapf2, float, a, float, s, return s+a; );
    FF_OCL_STENCIL_ELEMFUNC_ENV(mapf2, float, useless, i, A, float, S, return S[i]+A[i]; );
    ff_mapOCL_1D<Task2, oclTask2> oclMap2(t2, mapf2, &alloc); 
#else
    ff_mapOCL_1D<Task2, oclTask2> oclMap2(t2, "cl_code/reusek2.cl", &alloc); 
#endif
    SET_DEVICE_TYPE(oclMap2); 
    oclMap2.saveBinaryFile();  // save the compiled version in cl_code/oclMap.cl.bin
    oclMap2.reuseBinaryFile(); // if the binary file is present it is used

    if (oclMap2.run_and_wait_end()<0) {
        error("running oclMap2\n");
        return -1;
    }
    
#if defined(CHECK)
    for(size_t i=0; i<max_size; ++i) {
        if (S[i] != i/(float)(i*i+i+1)) {
            printf("Error %g != %g (i=%ld)\n", S[i], i/(float)(i*i+i+1), i);
            exit(1); //ctest
        }
    }
    printf("Result correct\n");   
#endif

    return 0;
}
