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
 *         (September 2015)       
 *  
 */

#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif

#include <vector>
#include <iostream>
#include <ff/map.hpp>
#include <ff/stencilReduceOCL.hpp>
#include <ff/selector.hpp>
#include <ff/taskf.hpp>

using namespace ff;

#define CHECK 1
#ifdef CHECK
#include "ctest.h"
#else
#define NACC 1
#endif

//#define SYNCHRONOUS_EXECUTION
const size_t DEFAULT_ARRAYSIZE=1024;


FF_OCL_MAP_ELEMFUNC(mapf, float, useless, i,
                    (void)useless;
                    return (i + 1/(i+1));
                    );


// this is the task
struct myTask {
    myTask(std::vector<float> &A):A(A) {}
    std::vector<float> &A;
};

// oclTask is used to transfer the input task to/from the OpenCL device
// - myTask is the type of the input task
// - float* is the type of OpenCL input array
// - float* is the type of OpenCL output array
struct oclTask: baseOCLTask<myTask, float, float> {
    void setTask(myTask *task) { 

        std::cout << "running on GPU\n";

        float *Aptr         = const_cast<float*>(task->A.data());
        const size_t Asize  = task->A.size();

        // A is not copied in input (only allocated), it will be deleted at the end
        setInPtr(Aptr, Asize, 
                 CopyFlags::DONTCOPY,
                 ReuseFlags::DONTREUSE,
                 ReleaseFlags::RELEASE);
        
        // A is copied back at the end, it will be deleted at the end
        setOutPtr(Aptr, Asize, 
                  CopyFlags::COPY,
                  ReuseFlags::DONTREUSE,
                  ReleaseFlags::RELEASE);
    }
};

// CPU map
struct cpuMap: ff_Map<myTask> {
    myTask *svc(myTask *in) {
        std::vector<float> &A = in->A;

        std::cout << "running on CPU\n";
        
        ff_Map::parallel_for(0,A.size(),[&A](const long i) {
                A[i] = (i + 1/(i+1));
            });    
        return in;
    }
};

// OpenCL map
struct oclMap: ff_mapOCL_1D<myTask, oclTask> {
    oclMap(const std::string &path, const std::string &kname): ff_mapOCL_1D<myTask, oclTask>(path, kname,nullptr,NACC) {}
    oclMap(): ff_mapOCL_1D<myTask, oclTask>(mapf,nullptr,NACC) {}
};

// selector, it decides whether to execute the CPU or OpenCL map
struct rprKernel: ff_nodeSelector<myTask> {
    rprKernel(int kernelId):kernelId(kernelId) {} 

    myTask *svc(myTask *in) {
	
        // make the decision 
        int selectedDevice = kernelId; 

        selectNode(selectedDevice);
        return ff_nodeSelector<myTask>::svc(in);
    }

    int kernelId;
};


int main(int argc, char *argv[]) {
    size_t size = DEFAULT_ARRAYSIZE;
    if (argc > 1) size = atol(argv[1]);
    
    std::vector<float> A0(size), A1(size);
    
    // creates the instances of the CPU map
    cpuMap cpumap0, cpumap1;
    // creates the instances of the GPU map
    oclMap oclmap0, oclmap1;

    // defines the 2 kernels providing the unique kernel id
    rprKernel kernel0(0); 
    rprKernel kernel1(1); 

    // adds the CPU and GPU instances to the kernel 0
    kernel0.addNode(cpumap0);
    kernel0.addNode(oclmap0);

    // adds the CPU and GPU instances to the kernel 1
    kernel1.addNode(cpumap1);
    kernel1.addNode(oclmap1);

    // executes initialization of each kernels
    if (kernel0.nodeInit()<0) {
        error("cannot initialize kernel0\n");
        return -1;
    }
    if (kernel1.nodeInit()<0) {
        error("cannot initialize kernel1\n");
        return -1;
    }
    
    // defines 2 lambdas that run the kernels
    auto F1 = [&]() { myTask task(A0); kernel0.svc(&task); };
    auto F2 = [&]() { myTask task(A1); kernel1.svc(&task); };

    // creates the asynchronous patterns with 2 executors
    ff_taskf taskf(2); 

    // adds the lambdas to the scheduler
    taskf.AddTask(F1); 
    taskf.AddTask(F2);

    // starts the execution of the scheduler waiting for the termination
    if (taskf.run_and_wait_end()<0) {
        error("error running taskf\n");
        return -1;
    }
        

#if defined(CHECK)
    bool wrong = false;
    for(size_t i=0;i<A0.size(); ++i) {
        if (A0[i] != (i + 1/(i+1))) {
            std::cerr << "Wrong result, A0[" << i <<"], expected " << (i+1/(i+1)) << " obtained "<< A0[i] << "\n";
            wrong = true;
        }
        if (A1[i] != (i + 1/(i+1))) {
            std::cerr << "Wrong result, A1[" << i <<"], expected " << (i+1/(i+1)) << " obtained "<< A1[i] << "\n";
            wrong = true;
        }

    }
    if (!wrong) std::cerr << "The result is OK\n";
    else exit(1); //ctest
#endif
    return 0;
}
