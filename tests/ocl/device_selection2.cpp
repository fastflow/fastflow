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
 * Author: Massimo Torquati, Marco Aldinucci
 * Date:   August 2015
 */

#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif

#define CHECK 1

#include <string>
#include <vector>
#include <ff/stencilReduceOCL.hpp>
//#include <ff/ocl/clEnvironment.hpp>

using namespace ff;

// the corresponding OpenCL type is in the (local) file 'ff_opencl_datatypes.cl'
struct mypair { float a; float b; };


FF_OCL_MAP_ELEMFUNC2(mapf, float, mypair, elem, 
                    return (elem.a * elem.b);
);

FF_OCL_STENCIL_COMBINATOR(reducef, float, (x), (y),
             return (x+y);
);


struct oclTask: public baseOCLTask<oclTask, mypair, float> {
    oclTask():M(NULL),Mout(NULL),result(0.0), size(0) {}
    oclTask(mypair *M, size_t size):M(M),Mout(NULL),result(0.0),size(size) {
        Mout = new float[size];
        assert(Mout);
    }
    ~oclTask() { if (Mout) delete [] Mout; }
    void setTask(const oclTask *t) { 
       assert(t);
       setInPtr(t->M, t->size);
       setOutPtr(t->Mout);
       setReduceVar(&(t->result));
     }

    mypair *M;
    float  *Mout, result;
    const size_t  size;
};

int main(int argc, char * argv[]) {
    size_t size = 1024;
    if (argc>1) size     =atol(argv[1]);
    printf("arraysize = %ld\n", size);

    mypair *M        = new mypair[size];
    for(size_t j=0;j<size;++j) {M[j].a=j*1.0; M[j].b=1; /*j*2.0;*/}

#if defined(CHECK)
    float r = 0.0;
    for(size_t j=0;j<size;++j) {
        r += M[j].a * M[j].b;
    }
#endif
    oclTask oclt(M, size);
    ff_mapReduceOCL_1D<oclTask> oclMR(oclt, mapf, reducef, 0.0);

    std::vector<std::string> res = oclMR.getDevicesInfo();
    
    for (size_t i=0; i<res.size(); ++i)
        std::cout << i << " - " << res[i] << std::endl;
    std::cout << "First CPU is " << oclMR.getCPUDevice() << "\n";
    std::cout << "First GPU is " << oclMR.getGPUDevice() << "\n";

    std::vector<ssize_t> allgpus = oclMR.getAllGPUDevices();
    for (size_t i=0; i<allgpus.size(); ++i)
        std::cout << "GPU #" << i << " ID " << allgpus[i] << std::endl;

    oclMR.pickGPU();
    
    oclMR.run_and_wait_end();

    //oclMR.willRunOnGPU();
    
    //oclMR.run_and_wait_end();


    delete [] M;
    printf("res=%.2f\n", oclt.result);
#if defined(CHECK)
    if (r != oclt.result) printf("Wrong result, should be %.2f\n", r);
    else printf("OK\n");
#endif
    return 0;
}
