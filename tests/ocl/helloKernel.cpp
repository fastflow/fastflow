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
 *         (August 2015)       
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


#if defined(BUILD_WITH_SOURCE)
const std::string kernel_path("cl_code/helloKernel.cl");

#else 

FF_OCL_MAP_ELEMFUNC(mapf, float, useless, i,
		    (void)useless;
		    return i + 1/(i+1);
		    );

#endif // BUILD_WITH_SOURCE


// this is the task
struct myTask {
    myTask(std::vector<float> A, const std::string &command):A(A),command(command) {}

    std::vector<float> A;
    const std::string command;
};


/* ---------------- helping function (naive implementation ------------------ */

typedef enum { IN, OUT } enum_t;
static std::tuple<CopyFlags, ReuseFlags, ReleaseFlags> 
parseCmd(int kernelid, enum_t direction, const std::string &cmd) {
    // here on the base of the command string, kernelid and direction we have to generate the proper tuple

    if (direction==IN) return std::make_tuple(CopyFlags::DONTCOPY,
                                              ReuseFlags::DONTREUSE,
                                              ReleaseFlags::RELEASE);
    return std::make_tuple(CopyFlags::COPY,
                           ReuseFlags::DONTREUSE,
                           ReleaseFlags::RELEASE);
}
static int parseCmd(int kernelId, const std::string &cmd) {
    return 1; // this is the oclMap
}
/* -------------------------------------------------------------------------- */


// oclTask is used to transfer the input task to/from the OpenCL device
// - myTask is the type of the input task
// - float* is the type of OpenCL input array
// - float* is the type of OpenCL output array
struct oclTask: baseOCLTask<myTask, float, float> {
    void setTask(myTask *task) { 
        float *Aptr         = const_cast<float*>(task->A.data());
        const size_t Asize  = task->A.size();
        const std::string &cmd(task->command);
        
        // define the parameter policy
        std::tuple<CopyFlags,ReuseFlags,ReleaseFlags> in   = parseCmd(0, IN, cmd); 
        std::tuple<CopyFlags,ReuseFlags,ReleaseFlags> out  = parseCmd(0, OUT, cmd);
        
        // A is not copied in input (false), nor the address is re-used (false), it will be deleted at the end (true) 
        setInPtr(Aptr, Asize, std::get<0>(in),std::get<1>(in), std::get<2>(in));
        
        // A is copied back at the end (true), the address is not re-used (false), it will be deleted at the end (true)
        setOutPtr(Aptr, Asize, std::get<0>(out), std::get<1>(out), std::get<2>(out));
    }
};

// CPU map
struct cpuMap: ff_Map<myTask> {
    myTask *svc(myTask *in) {
	std::vector<float> &A = in->A;

	ff_Map::parallel_for(0,A.size(),[&A](const long i) {
		A[i] = i + 1/(i+1);
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
        int selectedDevice = parseCmd(kernelId, in->command);
        
        in = reinterpret_cast<myTask*>(getNode(selectedDevice)->svc(in));
        return in;
    }

    int kernelId;
};


int main(int argc, char *argv[]) {
    size_t size = DEFAULT_ARRAYSIZE;
    if (argc > 1) size = atol(argv[1]);

    const std::string &command = "<kernel 0>; <platform:0:device:0>, <0:>; <0:RF>";

    std::vector<float> A(size);  // input and output array, in-place computation

    myTask task(std::move(A), command);

    cpuMap cpumap;
    oclMap oclmap;
    rprKernel kernel(0);  // 0 is the kernel id !!

    // on the base of the command, we have to select the device to use for the oclmap
    // oclmap.setDevices(....);

    kernel.addNode(cpumap);
    kernel.addNode(oclmap);

#if defined(SYNCHRONOUS_EXECUTION)    
    if (kernel.nodeInit()<0) {
	error("cannot initialize kernel\n");
	return -1;
    }

    kernel.svc(&task);
#else

#if 1
    if (kernel.nodeInit()<0) {
        error("cannot initialize kernel\n");
        return -1;
    }
    
    // this way to run the kernel asynchronously is convinient if we have multiple kernels
    // and a single sincronization point (a barrier)
    auto F = [&]() { kernel.svc(&task); };
    ff_taskf taskf(1); // just 1 thread in this case 
    taskf.AddTask(F);
    taskf.run();
    
    // let's do something else here

    taskf.wait(); // sync

#else
    // a simpler way 

    kernel.setTask(task);
    kernel.run(); 

    // let's do something else here

    kernel.wait();

#endif // if 0
#endif // SYNCHRONOUS_EXECUTION

#if defined(CHECK)
    bool wrong = false;
    for(size_t i=0;i<A.size(); ++i) 
	if (A[i] != (i + 1/(i+1))) {
	    std::cerr << "Wrong result, A[" << i <<"], expected " << (i+1/(i+1)) << " obtained "<< A[i] << "\n";
	    wrong = true;
	}
    if (!wrong) std::cerr << "The result is OK\n";
    else exit(1); //ctest
#endif
    return 0;
}
