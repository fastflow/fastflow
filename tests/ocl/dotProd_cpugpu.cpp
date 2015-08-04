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
/*  This program computes the dot-product of 2 arrays as:
 *
 *    s = sum_i(A[i]*B[i])
 *
*/

#include <iostream>
#include <iomanip>
#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif
#include <ff/farm.hpp>
#include <ff/map.hpp>
#include <ff/stencilReduceOCL.hpp>
#include <ff/selector.hpp>

using namespace ff;


FF_OCL_STENCIL_ELEMFUNC1(mapf, float, size, i, A, i_, float, B,
                         (void)size;
                         return A[i_] * B[i_];
                         );

FF_OCL_STENCIL_COMBINATOR(reducef, float, x, y,
                          return (x+y); 
                          );

struct Task: public baseOCLTask<Task, float, float> {
    Task():initVal(0.0),result(0.0),copy(true) {}
    Task(const std::vector<float> &A, const std::vector<float> &B):
        A(A),B(B), M(A.size()),result(0.0), copy(true) {
    }
    ~Task() {}
    void setTask(const Task *t) { 
       assert(t);
       
       // qui andrebbe interpretata la stringa dei comandi
       // ci vuole una fase post eleaborazione per fare le free ??


       setInPtr(const_cast<float*>(t->A.data()),  t->A.size());
       setEnvPtr(const_cast<float*>(t->B.data()), t->B.size(), t->copy);
       setOutPtr(const_cast<float*>(t->M.data()));

       setReduceVar(&(t->result));
     }

    void setResult(const float r) { result = r; }

    std::vector<float> A, B;
    std::vector<float> M;
    float initVal;
    float result;
    bool  copy;
};

struct Map: ff_Map<Task,Task,float> {

    Task *svc(Task *in) {
        std::vector<float> &A = in->A;
        std::vector<float> &B = in->B;
        const size_t arraySize = A.size();

        float sum = 0.0;
        ff_Map<Task,Task,float>::parallel_reduce(sum, 0.0F, 
                                                 0, arraySize, 
                                                 [&A,&B](const long i, float &sum) { sum += A[i] * B[i]; },
                                                 [](float &v, const float e) { v += e; });
        
        in->setResult(sum);
        return in;
    }
};


int main(int argc, char * argv[]) {    
    if (argc < 3) {
        std::cerr << "use: " << argv[0]  << " size 0|1\n";
        std::cerr << "    0: CPU\n";
        std::cerr << "    1: GPU\n";
        return -1;
    }
    size_t arraySize = atol(argv[1]);
    int cpugpu = atoi(argv[2]);
    assert(cpugpu == 0 || cpugpu == 1);
    int gpudev = clEnvironment::instance()->getCPUDevice(); // this exists for sure
    if (cpugpu != 0) {
        int tmp;
        if ((tmp = clEnvironment::instance()->getGPUDevice()) == -1) {
            std::cerr << "NO GPU device available, switching to CPU\n";
            cpugpu = 0;
        } else gpudev = tmp;
    }

    
    std::vector<float> A(arraySize);
    std::vector<float> B(arraySize);

    for(size_t i=0;i<arraySize;++i) {
        A[i] = 1.0; B[i]= i;
    }

    Map map;
    ff_mapReduceOCL_1D<Task> oclmap(mapf, reducef, 0.0);

    std::vector<cl_device_id> dev;
    dev.push_back(clEnvironment::instance()->getDevice(gpudev));
    oclmap.setDevices(dev);

    Task task(A,B);
    
    ff_nodeSelector<Task> selector(task);
    selector.addNode(map);
    selector.addNode(oclmap);

    selector.selectNode(cpugpu);
    selector.run_and_wait_end();

    std::cout << "Result = " << std::setprecision(10) << task.result << "\n";

    return 0;
}
