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

// REPARA code:    
//
//     initInput(A,S);
//
//     [[ rpr::kernel, rpr::map, rpr::in(A, k), rpr::out(A), rpr::target(CPU,GPU) ]]
//     for(size_t i=0;i<A.size(); ++i) {
//         A[i] /= (A[i]*A[i] + A[i] + 1);
//     }
//
//     [[ rpr::kernel, rpr::map, rpr::in(A,S), rpr::out(S), rpr::target(CPU,GPU) ]]
//     for(size_t i=0;i<A.size(); ++i) {
//       S[i] += A[i];
//

// to check the result
#define CHECK 1
#ifdef CHECK
#include "ctest.h"
#else
#define NACC 1
#endif

#include <ff/map.hpp>
#include <ff/selector.hpp>
#include <ff/stencilReduceOCL.hpp>
using namespace ff;

enum class Device { CPU=0, GPU=1}; 
static Device kernel1_device=Device::GPU, kernel2_device=Device::GPU;

/* -------------- First and second task types ----------- */
struct Task1 {
    Task1(std::vector<float> &A):A(A) {}
    std::vector<float> &A;
};
struct Task2 {
    Task2(std::vector<float> &A, std::vector<float> &S):A(A),S(S) {}
    std::vector<float> &A;
    std::vector<float> &S;
};

/* --------- OpenCL interface object task types --------- */
struct oclTask1: public baseOCLTask<Task1, float> {
    oclTask1() {}
    void setTask(Task1 *t) { 

        MemoryFlags mfin(CopyFlags::COPY,  ReuseFlags::DONTREUSE, ReleaseFlags::DONTRELEASE);
        MemoryFlags mfout(CopyFlags::COPY, ReuseFlags::DONTREUSE, ReleaseFlags::DONTRELEASE);

        if (kernel1_device == Device::GPU && kernel2_device == kernel1_device) 
            mfout.copy=CopyFlags::DONTCOPY;

        setInPtr(t->A.data(), t->A.size(),  mfin);  // set host input pointer
        setOutPtr(t->A.data(), t->A.size(), mfout); // set host output pointer
    }
};

struct oclTask2: public baseOCLTask<Task2, float> {
    oclTask2() {}
    void setTask(Task2 *t) { 
        MemoryFlags mfin(CopyFlags::COPY,  ReuseFlags::DONTREUSE, ReleaseFlags::DONTRELEASE);
        MemoryFlags mfout(CopyFlags::COPY, ReuseFlags::DONTREUSE, ReleaseFlags::DONTRELEASE);
        
        if (kernel1_device == Device::GPU && kernel2_device == kernel1_device) {
            mfin.copy=CopyFlags::DONTCOPY, mfin.reuse=ReuseFlags::REUSE;
        }

        setInPtr(t->A.data(), t->A.size(),  mfin);  // set host input pointer
        setEnvPtr(t->S.data(), t->S.size(), mfout); // set host env pointer
        setOutPtr(t->S.data(), t->S.size(), mfout); // set host output pointer
    }
};

/* ------------ CPU-based first and second map --------- */
struct Map1: ff_Map<Task1> {
    using map = ff_Map<Task1>;
    Map1():ff_Map<Task1>(ff_realNumCores()) {}

    Task1 *svc(Task1 *in) {
	std::vector<float> &A = in->A;
	map::parallel_for(0,A.size(),[&](const long i) {
		A[i] /= (A[i]*A[i] + A[i] + 1);
	    });
	
	return in;
    }
};
struct Map2: ff_Map<Task2> {
    using map = ff_Map<Task2>;
    Map2():ff_Map<Task2>(ff_realNumCores()) {}

    Task2 *svc(Task2 *in) {
	std::vector<float> &A = in->A;
	std::vector<float> &S = in->S;
	map::parallel_for(0, S.size(), [&](const long i) {
		S[i] += A[i];
	    });	
	return in;
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
    if (argc != 3) {
        std::cerr << "use: " << argv[0] 
                  << " 0|1 0|1\n";
        std::cerr << " 0: execution on CPU cores\n";
        std::cerr << " 1: execution on the local GPU\n";
        return -1;
    }
    kernel1_device = (std::stoi(argv[1])==0 ? Device::CPU:Device::GPU);
    kernel2_device = (std::stoi(argv[2])==0 ? Device::CPU:Device::GPU);

    const size_t max_size   = 1000;

    std::vector<float> S(max_size);
    std::vector<float> A(max_size);
    
    // init data
    initInput(A,S);
    
    // OpenCL device shared allocator
    ff_oclallocator alloc;

    // instances of the first and second task
    Task1 t1(A);    
    Task2 t2(A,S);

    // ------ OpenCL map1 instance 
    FF_OCL_MAP_ELEMFUNC_1D(mapf1, float, a, 
			   return a / (a*a + a + 1);
			   );    
    ff_mapOCL_1D<Task1, oclTask1> oclMap1(mapf1, &alloc); 
    SET_DEVICE_TYPE(oclMap1);  
    //oclMap1.pickGPU();

    // ------ CPU map1
    Map1  map1;

    // ------ selector1 instance
    ff_nodeSelector<Task1> selector1(t1);
    selector1.addNode(map1);
    selector1.addNode(oclMap1);

    // ------ selecting where to execute the kernel (CPU)
    selector1.selectNode(kernel1_device == Device::GPU ? 1:0);
    
    if (selector1.run_and_wait_end()<0) {
	error("running selector1\n");
	return -1;
    }

    // ------ OpenCL map2 instance 
    FF_OCL_MAP_ELEMFUNC_1D_ENV(mapf2, float, a, float, s,
			       return s+a;
			       );
    ff_mapOCL_1D<Task2, oclTask2> oclMap2(mapf2, &alloc); 
    SET_DEVICE_TYPE(oclMap2); 
    //oclMap2.pickGPU();

    // ------ CPU map2 instance
    Map2 map2;

    // ------ selector2  instance 
    ff_nodeSelector<Task2> selector2(t2);
    selector2.addNode(map2);
    selector2.addNode(oclMap2);

    // ------ selecting where to execute the kernel (GPU)
    selector2.selectNode(kernel2_device == Device::GPU ? 1:0);

    if (selector2.run_and_wait_end()<0) {
	error("running selector1\n");
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
