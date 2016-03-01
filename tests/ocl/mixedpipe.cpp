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
 *  Author: Massimo Torquati (August 2015)
 *
 */

#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif

#include <vector>
#include <iostream>

#include <ff/farm.hpp>
#include <ff/map.hpp>
#include <ff/stencilReduceOCL.hpp>
#include <ff/selector.hpp>
#include <math.h>

//#define SYSTEM_HAS_GPU 1

using namespace ff;

#define CHECK 1
#ifdef CHECK
#include "ctest.h"
#else
#define NACC 1
#endif

#define DEFAULT_ARRAYSIZE 1024
#define DEFAULT_NVECTORS 2048
#define DEFAULT_COMMAND "1:1:1"

#if defined(BUILD_WITH_SOURCE)

const std::string path_k1("cl_code/mixedpipe_k1.cl");
const std::string path_k2("cl_code/mixedpipe_k2.cl");
const std::string path_k3("cl_code/mixedpipe_k3.cl");

#else 

FF_OCL_STENCIL_ELEMFUNC_ENV(map1f, float, useless, i, in, int, k_,
                         (void)useless; const int k = *k_;
                         return (float)((k+1) + i);
                         );
FF_OCL_STENCIL_ELEMFUNC_ENV(map2f, float, useless, i, A, float, B,
                         (void)useless;
                         return A[i] * B[i];
                         );
FF_OCL_STENCIL_ELEMFUNC_2ENV(map3f, float, useless, i, R, float, A, float, sum_,
                         (void)useless; const float sum = *sum_;
                         return R[i] + 1 / (A[i] + sum);
                         );

//implicit input
//FF_OCL_STENCIL_ELEMFUNC_1D_ENV(map1f, float, N, i, int,
//	return (float)(GET_ENV(0) + i + 1);
//);
//
//FF_OCL_STENCIL_ELEMFUNC_1D_ENV(map2f, float, N, i, float,
//	return GET_IN(i) * GET_ENV(i);
//);
//
//FF_OCL_STENCIL_ELEMFUNC_1D_2ENV(map3f, float, N, i, float, float,
//	return GET_IN(i) + (float)1 / (GET_ENV1(i) + GET_ENV2(0));
//);

FF_OCL_REDUCE_COMBINATOR(reducef, float, x, y,
                          return (x+y); 
                          );
#endif


/* This is the stream type. Each element of the stream is a Task.
 *
 */
struct Task: public baseOCLTask<Task, float, float> {
	float combinator(float const &x, float const &y) {return x+y;}

    Task():sum(0.0),arraySize(0),k(0),kernelId(0),C(nullptr),R(nullptr) {}
    Task(const size_t size, size_t k):
    	A(size),sum(0.0),arraySize(size), k(k),kernelId(0),C(nullptr),R(nullptr) {}

    void setTask(Task *t) { 
       assert(t);

       switch (t->kernelId) {
       case 1: {
           setInPtr(const_cast<float*>(t->A.data()), t->A.size(), 
                    CopyFlags::DONTCOPY);                                // A not copied 
           setEnvPtr(&(t->k), 1);                                        // k always copied
           setOutPtr(const_cast<float*>(t->A.data()), t->A.size());      // A received back
       } break;                                                          // A kept (by default)
       case 2: {
           setInPtr(const_cast<float*>(t->A.data()),  t->A.size());      // A copied
           setEnvPtr(const_cast<float*>(t->C->data()), t->C->size(), 
                     (t->k==0)?CopyFlags::COPY:CopyFlags::DONTCOPY);   // C copied only the 1st time
           setReduceVar(&(t->sum));                                      // sum 
       } break;                                                          // A,C kept (by default)
       case 3: {
           setInPtr(const_cast<float*>(t->R->data()),  t->R->size(),     // R copied only the 1st time
                    (t->k==0)?CopyFlags::COPY:CopyFlags::DONTCOPY);    // 
           setEnvPtr(const_cast<float*>(t->A.data()),  t->A.size());     // A copied
           setEnvPtr(&(t->sum), 1);                                      // sum
           setOutPtr(const_cast<float*>(t->R->data()),  t->R->size());   // R get back result
       } break;                                                          // R kept (by default)
       default: abort();
       }
     }

    /* stream items */
    std::vector<float> A;
    float sum;

    /* ------- */
    const size_t arraySize;   
    const size_t k;           // the stream is generated by a for loop, this is the index
    size_t kernelId;
    std::vector<float> *C;
    std::vector<float> *R;

};

/* 
 * Kernel class, it dynamically selects the logical device where the kernel has to be executed.
 *  0 is the first logical device, in this test the C++ Map (based on the parallel_for)
 *  1 is the second logical device, in this test the OpenCL Map 
 *
 * The OpenCL Map can be executed either on the CPUs or on the GPU (if any) depending on the 
 * SET_DEVICE_TYPE setting.
 *
 *
 */
struct Kernel: ff_nodeSelector<Task> {
    using selector = ff_nodeSelector<Task>;

    // simple function to select the device id from the command string
    int getDeviceId() {
        std::string cmd(command);
        int n = -1;
        for(size_t i=0;i<kernelId;++i) { // kernelId is > 0
            cmd = cmd.substr(n+1);
            n = cmd.find_first_of(":");
            if (n == -1) n = cmd.length();
        }
        return atoi(cmd.substr(0,n).c_str());
    }
    
    Kernel(const size_t kernelId, std::string command, 
           const size_t NVECTORS, const size_t arraySize,
           std::vector<float> &C, std::vector<float> &R):
        kernelId(kernelId),command(command),NVECTORS(NVECTORS),arraySize(arraySize),C(&C),R(&R) {}

    Task *svc(Task *in) {
        int selectedDevice = getDeviceId();  // select the device id from the command string        
        // set the selected device
        selectNode(selectedDevice);

        switch (kernelId) {
        case 1: {
            assert(in == nullptr);
            for(size_t k=0;k<NVECTORS;++k) {
                Task *task = new Task(arraySize, k);
                task->kernelId = 1;
                task = selector::svc(task);
                ff_send_out(task);
            }
            return EOS;                
        } break;
        case 2: {
            in->C = C;                
            in->kernelId = 2;
            in = selector::svc(in);
            return in;            
        } break;
        case 3: {
            in->R = R;                
            in->kernelId = 3;
            in = selector::svc(in);
            delete in;
            return GO_ON;         
        } break;
        default: abort();
        }
        return EOS;
    }
    
    // ---------  configuration value ----------
    const size_t      kernelId;
    const std::string command;
    const size_t      NVECTORS;
    const size_t      arraySize;

    // these are not part of the stream, they are data needed to compute the kernel 
    // (both kernel2 and kernel3)
    std::vector<float> *C;
    std::vector<float> *R;
};


/* ------------------ CPU map and map-reduce ------------------ */
struct Map_kernel1: ff_Map<Task,Task,float> {
    Task *svc(Task *task) {
        const size_t arraySize = task->arraySize;
        const size_t &k        = task->k;
        std::vector<float> &A  = task->A;

        /* ------------ user code ------------- */
        for(size_t i=0;i<arraySize;++i) 
            A[i] = (k+1)+i; 
        /* ------------------------------------ */
        
        return task;
    }
};
struct Map_kernel2: ff_Map<Task,Task,float> {
    Task *svc(Task *task) {
        const size_t arraySize = task->arraySize;
        std::vector<float> &A  = task->A;
        std::vector<float> &C  = *(task->C); 
        float &sum             = task->sum;

        /* ------------ user code ------------- */
        sum = 0;
        for(size_t i=0;i<arraySize;++i)
            sum += A[i] * C[i];
        /* ------------------------------------ */

        return task;
    }
};
struct Map_kernel3: ff_Map<Task,Task,float> {
    Task *svc(Task *task) {
        const size_t arraySize = task->arraySize;
        std::vector<float> &A  = task->A;
        float &sum             = task->sum;
        std::vector<float> &R  = *(task->R); 

        /* ------------ user code ------------- */
        for(size_t i=0;i<arraySize;++i)
            R[i] += 1 / (A[i] + sum);
        /* ------------------------------------ */

        return task;
    }
};
/* ------------------------------------------------------------ */

int main(int argc, char * argv[]) {

    std::cerr << "use: " << argv[0]  << " size nvectors command-string\n";
    std::cerr << " command-string (example) : \"0:1:0\"\n";
    std::cerr << "    first kernel on device  0\n";
    std::cerr << "    second kernel on device 1\n";
    std::cerr << "    third kernel on device  0\n";

    size_t arraySize = DEFAULT_ARRAYSIZE;
    size_t NVECTORS = DEFAULT_NVECTORS;
    std::string command(DEFAULT_COMMAND);

	if (argc > 1) {
		arraySize = atol(argv[1]);
		if (argc > 2) {
			NVECTORS = atol(argv[2]);
			if (argc > 3) {
				command = std::string(argv[3]);
			}
		}
	}

    std::vector<float> A(arraySize);
    std::vector<float> C(arraySize);    
    std::vector<float> R(arraySize);    

    for(size_t i=0;i<arraySize;++i) {
        C[i] = i;
        R[i] = 0.0;
    }

    // CPU map and map-reduce
    Map_kernel1 map1;
    Map_kernel2 map2;
    Map_kernel3 map3;

    // GPU map and map-reduce
#if defined(BUILD_WITH_SOURCE)
    ff_mapOCL_1D<Task>       oclmap1(path_k1, "map1f", nullptr, NACC);             
    ff_mapReduceOCL_1D<Task> oclmap2(path_k2, "map2f", "reducef",0.0,nullptr,NACC);   
    ff_mapOCL_1D<Task>       oclmap3(path_k3, "map3f", nullptr, NACC);     
    oclmap1.saveBinaryFile(); oclmap1.reuseBinaryFile();
    oclmap2.saveBinaryFile(); oclmap2.reuseBinaryFile();
    oclmap3.saveBinaryFile(); oclmap3.reuseBinaryFile();    
#else
    ff_mapOCL_1D<Task>       oclmap1(map1f, nullptr, NACC);             
    ff_mapReduceOCL_1D<Task> oclmap2(map2f,reducef,0.0,nullptr,NACC);   
    ff_mapOCL_1D<Task>       oclmap3(map3f, nullptr, NACC);             
#endif

    SET_DEVICE_TYPE(oclmap1);
    SET_DEVICE_TYPE(oclmap2);
    SET_DEVICE_TYPE(oclmap3);
    
    Kernel k1(1,command, NVECTORS,arraySize, C, R);
    Kernel k2(2,command, NVECTORS,arraySize, C, R);
    Kernel k3(3,command, NVECTORS,arraySize, C, R);

    k1.addNode(map1);
    k1.addNode(oclmap1);
    k2.addNode(map2);
    k2.addNode(oclmap2);
    k3.addNode(map3);
    k3.addNode(oclmap3);

    ff_Pipe<> pipe(k1,k2,k3);
    if (pipe.run_and_wait_end()<0) {
        error("pipeline");
        return -1;
    }

#if defined(CHECK)
    {
        std::vector<float> A(arraySize);
        std::vector<float> C(arraySize);    
        std::vector<float> _R(arraySize);    
        
        for(size_t i=0;i<arraySize;++i) {
            C[i] = i;
            _R[i] = 0.0;
        }
        
        // sequential code annotated with REPARA attributes 
        //[[ rpr::pipeline, rpr::stream(A,sum) ]]
        for(size_t k=0;k<NVECTORS;++k) {
            
            //[[rpr::kernel, rpr::in(A,k), rpr::out(A), rpr::target(CPU,GPU)]]
            for(size_t i=0;i<arraySize;++i) 
                A[i] = (k+1)+i; 
            
            float sum = 0;
            //[[rpr::kernel, rpr::in(sum, A,C), rpr::out(sum), rpr::reduce(sum, +), rpr::target(CPU,GPU)]]
            for(size_t i=0;i<arraySize;++i)
                sum += A[i] * C[i];
            
            //[[rpr::kernel, rpr::in(R,A,sum), rpr::out(R), rpr::target(CPU,GPU)]]
            for(size_t i=0;i<arraySize;++i)
                _R[i] += 1 / (A[i] + sum);
        }

        bool wrong = false;
        for(size_t i=0;i<arraySize;++i)
            if (std::abs(R[i] -_R[i]) > 0.0001) {
                std::cerr << "Wrong result " <<  R[i] << " should be " << _R[i] << "\n";
                wrong = true;
            }
        if (!wrong) std::cerr << "OK!\n";
        else exit(1); //ctest
    }
#else
    std::cerr << "Result not checked\n";
#endif
    return 0;
}
