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

// TO BE COMPLETED


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

FF_OCL_STENCIL_ELEMFUNC1(map1f, float, useless, i, in, i_, int, k_,
                         (void)useless; const int k = *k_;

                         return (float)((k+1) + i_);
                         );
FF_OCL_STENCIL_ELEMFUNC1(map2f, float, useless, i, A, i_, float, B,
                         (void)useless;

                         return A[i_] * B[i_];
                         );
FF_OCL_STENCIL_ELEMFUNC2(map3f, float, useless, i, R, i_, float, A, float, sum_,
                         (void)useless; const float sum = *sum_;

                         return R[i_] + (1 / (A[i_] + sum));
                         );
FF_OCL_STENCIL_COMBINATOR(reducef, float, x, y,

                          return (x+y); 
                          );

struct Task: public baseOCLTask<Task, float, float> {
    Task():arraySize(0),k(0),sum(0.0) {}
    Task(const size_t size, size_t k):arraySize(size), k(k), A(size),sum(0.0) {}

    void setTask(const Task *t) { 
       assert(t);

       switch (t->kernelId) {
       case 1: {
           setInPtr(const_cast<float*>(t->A.data()),  t->A.size());
           setEnvPtr(&(t->k), 1);
       } break;
       case 2: {
           setInPtr(const_cast<float*>(t->A.data()),  t->A.size());
           setEnvPtr(const_cast<float*>(t->C->data()), t->C->size());
           setReduceVar(&(t->sum));           
       } break;
       case 3: {
           setInPtr(const_cast<float*>(t->A.data()),  t->A.size());
           setEnvPtr(&(t->sum), 1);
           setOutPtr(const_cast<float*>(t->R->data()),  t->R->size());
       } break;
       default: abort();
       }
     }

    const size_t arraySize;
    size_t k;

    /* stream items */
    std::vector<float> A;
    float sum;

    /* ------- */
    size_t kernelId;
    std::vector<float> *C;
    std::vector<float> *R;

};

struct Kernel: ff_nodeSelector<Task> {
    Kernel(const size_t kernelId, std::string command, 
           const size_t NVECTORS, const size_t arraySize,
           std::vector<float> &C, std::vector<float> &R):
        kernelId(kernelId),command(command),NVECTORS(NVECTORS),arraySize(arraySize),C(&C),R(&R) {}

    Task *svc(Task *in) {
        int selectedDevice = 1;
        
        switch (kernelId) {
        case 1: {
            assert(in == nullptr);
            for(size_t k=0;k<NVECTORS;++k) {
                Task *task = new Task(arraySize, k);
                task->kernelId = 1;
                //task = reinterpret_cast<Task*>(getNode(selectedDevice)->svc(task));
                task = reinterpret_cast<Task*>(getNode(0)->svc(task));
                ff_send_out(task);
            }
            return EOS;                
        } break;
        case 2: {
            in->C = C;
            in->kernelId = 2;
            in = reinterpret_cast<Task*>(getNode(selectedDevice)->svc(in));
            return in;            
        } break;
        case 3: {
            in->R = R;
            in->kernelId = 3;
            //in = reinterpret_cast<Task*>(getNode(selectedDevice)->svc(in));
            in = reinterpret_cast<Task*>(getNode(0)->svc(in));
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

    std::vector<float> *C;
    std::vector<float> *R;
};


/* ------------------ CPU map and map-reduce ------------------ */
struct Map_kernel1: ff_Map<Task,Task,float> {
    Task *svc(Task *task) {
        const size_t arraySize = task->arraySize;
        const size_t &k        = task->k;
        std::vector<float> &A  = task->A;
        for(size_t i=0;i<arraySize;++i) 
            A[i] = (k+1)+i; 
        
        return task;
    }
};
struct Map_kernel2: ff_Map<Task,Task,float> {
    Task *svc(Task *task) {
        const size_t arraySize = task->arraySize;
        std::vector<float> &A  = task->A;
        std::vector<float> &C  = *(task->C);    // FIX
        float &sum             = task->sum;

        sum = 0;
        for(size_t i=0;i<arraySize;++i)
            sum += A[i] * C[i];

        return task;
    }
};
struct Map_kernel3: ff_Map<Task,Task,float> {
    Task *svc(Task *task) {
        const size_t arraySize = task->arraySize;
        std::vector<float> &A  = task->A;
        float &sum             = task->sum;
        std::vector<float> &R  = *(task->R);   // FIX

        for(size_t i=0;i<arraySize;++i)
            R[i] += 1 / (A[i] + sum);

        return GO_ON;
    }
};
/* ------------------------------------------------------------ */

int main(int argc, char * argv[]) {    
    if (argc < 4) {
        std::cerr << "use: " << argv[0]  << " size nvectors command-string\n";
        return -1;
    }
    size_t arraySize = atol(argv[1]);
    size_t NVECTORS  = atol(argv[2]);
    std::string command(argv[3]);

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
    ff_mapOCL_1D<Task>       oclmap1(map1f);             oclmap1.runOnCPU();
    ff_mapReduceOCL_1D<Task> oclmap2(map2f,reducef,0.0); oclmap2.runOnCPU();
    ff_mapOCL_1D<Task>       oclmap3(map3f);             oclmap3.runOnCPU();
    
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
    // NOTE FOR OpenCL:  TODO
    //     - kernel0
    //          A only allocated on the device, no copy
    //          k copied
    //          A received back
    //          A keep
    //
    //     - kernel1
    //          A neither allocated nor copied, previous A should be used
    //          C if (k == 0)  created and copied on the device
    //            else         nothing (re-use the previous one)
    //        sum allocated, not copied and received back
    //          A keep
    //          C keep
    //
    //     - kernel2
    //        sum copy
    //          A neither allocated nor copied, previous A should be used
    //          R if (k == 0) created and copied on the device
    //            else        nothing (re-use the previous one)
    //          R received back
    //          A deleted
    //          R keep
    {
        std::vector<float> A(arraySize);
        std::vector<float> C(arraySize);    
        std::vector<float> _R(arraySize);    
        
        for(size_t i=0;i<arraySize;++i) {
            C[i] = i;
            _R[i] = 0.0;
        }
        
        // sequential code annotated with REPARA attributes 
        for(size_t k=0;k<NVECTORS;++k) {
            
            //[[rpr::kernel rpr::in(A) rpr::out(A) rpr::target(CPU,GPU)]]
            for(size_t i=0;i<arraySize;++i) 
                A[i] = (k+1)+i; 
            
            float sum = 0;
            //[[rpr::kernel rpr::in(sum, A,C) rpr::out(sum)  rpr::reduce(sum, +) rpr::target(CPU,GPU)]]
            for(size_t i=0;i<arraySize;++i)
                sum += A[i] * C[i];
            
            //[[rpr::kernel rpr::in(A,sum) rpr::out(R) rpr::target(CPU,GPU)]]
            for(size_t i=0;i<arraySize;++i)
                _R[i] += 1 / (A[i] + sum);
        }

        bool wrong = false;
        for(size_t i=0;i<arraySize;++i)
            if (R[i] != _R[i]) {
                std::cerr << "Wrong result " <<  R[i] << " should be " << _R[i] << "\n";
                wrong = true;
            }
        if (!wrong) std::cerr << "OK!\n";
    }
#endif
        
    

    return 0;
}
