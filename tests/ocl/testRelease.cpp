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
/* 
 * This simple example shows how to use the "setTask" and "releaseTask" methods
 * when input data to the OpenCL node has a non-contiguous memory layout.
 * 
 */


#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif

#include <vector>
#include <iostream>

#include <ff/pipeline.hpp>
#include <ff/stencilReduceOCL.hpp>
#include <math.h>

using namespace ff;

#define CHECK 1
#ifdef CHECK
#include "ctest.h"
#else
#define NACC 1
#endif


FF_OCL_MAP_ELEMFUNC(mapf, float, elem, idx,  return elem + idx );

// default parameters
const size_t N            = 20;
const size_t M            = 30;
const size_t streamLength = 10;

struct myTask {
    myTask(const size_t N, const size_t M, 
           std::vector<std::vector<float> > &A):N(N),M(M),A(A) {}
    myTask(const size_t N, const size_t M):N(N),M(M) {}

    const size_t N;
    const size_t M;
    std::vector<std::vector<float> > A;  // non-contigous memory layout

    // std::string command;
};


struct oclTask: baseOCLTask<myTask, float> {
    void setTask(myTask *task) { 
        const size_t N = task->N;
        const size_t M = task->M;
        const size_t size = N * M;

        // allocate a local buffer of correct size
        buffer = new float[size];
        // copy the data into the buffer
        for(size_t i=0;i<N;++i)
            for(size_t j=0;j<M;++j)
                buffer[i*M+j] = task->A[i][j];
     
        // set host input and output pointers to the OpenCL device
        setInPtr(buffer, size);
        setOutPtr(buffer, size);
    }

    // this method is called when the device computation is finished
    void releaseTask(myTask *task) {
        const size_t N = task->N;
        const size_t M = task->M;
        std::vector<std::vector<float> > &A = task->A;

        // copy back the device output data into A
        for(size_t i=0;i<N;++i)
            for(size_t j=0;j<M;++j)
                A[i][j] = buffer[i*M+j];

        // remove the local buffer
        delete [] buffer;
    }

    float *buffer;
};

// first stage of the pipeline. It generates 'slen' myTask objects
struct First: ff_node_t<myTask> {
    First(const size_t N, const size_t M, const size_t slen):N(N),M(M),slen(slen) {}
    myTask *svc(myTask*) {
        for(size_t k=0;k<slen;++k) {
            myTask *t = new myTask(N,M);
            t->A.resize(N);
            for(size_t i=0;i<N;++i) {
                t->A[i].resize(M);
                for(size_t j=0;j<M;++j)
                    t->A[i][j] = k+i+j;
            }
            ff_send_out(t);
        }
        return EOS;
    }

    const size_t N,M,slen;
};

// last stage of the pipeline. It simply gathers the stream elements and checks correctness.
struct Last: ff_node_t<myTask> {
    myTask *svc(myTask *task) {
#if defined(CHECK)
        static size_t counter = 0;
        bool wrong = false;
        const size_t N = task->N;
        const size_t M = task->M;
        const std::vector<std::vector<float> > &A = task->A;
        for(size_t i=0;i<N;++i) {
            for(size_t j=0;j<M;++j)
                if (A[i][j] != counter + (i+j + i*M+j)) {
                    std::cerr << "Wrong value (" << i << ","<<j<<"), expected " << (i+j + i*M+j) << " obtained " << A[i][j] << "\n";
                    wrong = true;
                }
        }
        if (!wrong) std::cerr << "OK!\n";
        else exit(1); //ctest
        ++counter;
#endif
        return GO_ON;
    }
};


int main(int argc, char * argv[]) {

    First first(N,M, streamLength);
    ff_mapOCL_1D<myTask, oclTask> mapocl(mapf);
    Last last;
    ff_Pipe<> pipe(first, mapocl, last);
    if (pipe.run_and_wait_end()<0) {
        error("pipeline");
        return -1;
    }

    return 0;
}
