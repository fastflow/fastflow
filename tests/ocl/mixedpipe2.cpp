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
/* This is another version of the mixedpipe examples. In this case the external for loop (for k) 
 * is not transformed in a parallel pipeline computation. 
 * The 3 kernels contained in the for loop body are executed in sequence optimizing the  
 * space on the GPU device (if GPU is selected).
 *
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

FF_OCL_REDUCE_COMBINATOR(reducef, float, x, y, return (x+y));

struct Task: public baseOCLTask<Task, float> {
	float combinator(float const &x, float const &y) {return x+y;}

    Task():sum(0.0),arraySize(0),k(0),kernelId(0),C(nullptr),R(nullptr) {}
    Task(const size_t size, size_t k):
    	A(size),sum(0.0),arraySize(size),k(k),kernelId(0),C(nullptr),R(nullptr) {}
    
    void setTask(Task *t) { 
        assert(t);
        
        switch (t->kernelId) {
        case 1: {
            setInPtr(const_cast<float*>(t->A.data()), t->A.size(),CopyFlags::DONTCOPY);      // A not copied 
            setEnvPtr(&(t->k), 1);                                                           // k always copied
            setOutPtr(const_cast<float*>(t->A.data()), t->A.size(),CopyFlags::DONTCOPY);     // A not received back
        } break;                                                                             // A kept
        case 2: {
            setInPtr(const_cast<float*>(t->A.data()),  t->A.size(), 
                     CopyFlags::DONTCOPY, ReuseFlags::REUSE);                               // A not copied, reusing the previous buffer
            setEnvPtr(const_cast<float*>(t->C->data()), t->C->size(), 
                      CopyFlags::COPY,ReuseFlags::DONTREUSE,ReleaseFlags::RELEASE);       // C copied and then released
            setReduceVar(&(t->sum));                                                         // sum 
        } break;                                                                             // A kept (by default)
        case 3: {
            setInPtr(const_cast<float*>(t->R->data()), t->R->size(), 
                     (t->k==0)?CopyFlags::COPY:CopyFlags::DONTCOPY);                       // R copied only the 1st time
            setEnvPtr(const_cast<float*>(t->A.data()),  t->A.size(), 
                      CopyFlags::DONTCOPY, ReuseFlags::REUSE);                                // A not copied, reusing the previous buffer (*)
            setEnvPtr(&(t->sum), 1);                                                         // sum
            setOutPtr(const_cast<float*>(t->R->data()),  t->R->size());                      // R get back result
        } break;                                                                             // R kept (by default)
        default: abort();
        }
    } // (*): note that A cannot be release here because kernel3 is re-using the buffer allocated in the first kernel !    


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

/* Kernel class, it selects the device on which to run the kernel code.
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
            in->kernelId = 1;
            in = selector::svc(in);
            return in;                
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

            // no delete here

            return in;            
        } break;
        default: abort();
        }
        return EOS;
    }
    
    const size_t      kernelId;
    const std::string command;
    const size_t      NVECTORS;
    const size_t      arraySize;

    // these are not part of the stream, they are data neede to compute the kernel (kernel2 and kernel3)
    std::vector<float> *C;
    std::vector<float> *R;
};


/* ------------------ CPU map and map-reduce ------------------ */
struct Map_kernel1: ff_Map<Task,Task,float> {
    Task *svc(Task *task) {
        const size_t arraySize = task->arraySize;
        const size_t &k        = task->k;
        std::vector<float> &A  = task->A;

        /* ------------ use code ------------- */
        for(size_t i=0;i<arraySize;++i) 
            A[i] = (k+1)+i; 
        /* ----------------------------------- */
        
        return task;
    }
};
struct Map_kernel2: ff_Map<Task,Task,float> {
    Task *svc(Task *task) {
        const size_t arraySize = task->arraySize;
        std::vector<float> &A  = task->A;
        std::vector<float> &C  = *(task->C); 
        float &sum             = task->sum;

        /* ------------ use code ------------- */
        sum = 0;
        for(size_t i=0;i<arraySize;++i)
            sum += A[i] * C[i];
        /* ----------------------------------- */

        return task;
    }
};
struct Map_kernel3: ff_Map<Task,Task,float> {
    Task *svc(Task *task) {
        const size_t arraySize = task->arraySize;
        std::vector<float> &A  = task->A;
        float &sum             = task->sum;
        std::vector<float> &R  = *(task->R); 

        /* ------------ use code ------------- */
        for(size_t i=0;i<arraySize;++i)
            R[i] += 1 / (A[i] + sum);
        /* ----------------------------------- */

        return task;
    }
};
/* ------------------------------------------------------------ */

int main(int argc, char * argv[]) {    
	//if (argc < 4) {
	std::cerr << "use: " << argv[0] << " size nvectors command-string\n";
	std::cerr << " command-string (example) : \"0:1:0\"\n";
	std::cerr << "    first kernel on device  0\n";
	std::cerr << "    second kernel on device 1\n";
	std::cerr << "    third kernel on device  0\n";
	//return -1;
	//}
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

    if (command != "0:0:0" && command != "1:1:1") {
        std::cerr << "Wrong command, for this test the only valid commands are 0:0:0, 1:1:1\n";
        return -1;
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

    // we have to use the same OpenCL allocator for each kernel
    ff_oclallocator allocator;

    // GPU map and map-reduce
    ff_mapOCL_1D<Task>       oclmap1(map1f,&allocator,NACC);             SET_DEVICE_TYPE(oclmap1);
    ff_mapReduceOCL_1D<Task> oclmap2(map2f,reducef,0.0,&allocator,NACC); SET_DEVICE_TYPE(oclmap2);
    ff_mapOCL_1D<Task>       oclmap3(map3f,&allocator,NACC);             SET_DEVICE_TYPE(oclmap3);
    
    Kernel k1(1,command, NVECTORS,arraySize, C, R);
    Kernel k2(2,command, NVECTORS,arraySize, C, R);
    Kernel k3(3,command, NVECTORS,arraySize, C, R);

    k1.addNode(map1);
    k1.addNode(oclmap1);
    k2.addNode(map2);
    k2.addNode(oclmap2);
    k3.addNode(map3);
    k3.addNode(oclmap3);

    // needed because threads are not started so svc_init is not called 
    k1.nodeInit();
    k2.nodeInit();
    k3.nodeInit();

    for(size_t k=0;k<NVECTORS;++k) {
        Task task(arraySize, k);
        Task *out = k1.svc(&task);
        out       = k2.svc(out);
        out       = k3.svc(out);
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
        // NO PIPELINE HERE
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
            if (std::abs(R[i] - _R[i]) > 0.0001) {
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
