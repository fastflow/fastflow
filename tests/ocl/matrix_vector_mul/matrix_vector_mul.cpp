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
 * Author: Massimo Torquati <torquati@di.unipi.it> 
 * Date:   October 2014
 */
#include <cassert>
#include <iostream>
#include <ff/mapOCL.hpp>

using namespace ff;

/* --------------- OpenCL code ------------------- */
FF_ARRAY2_CENV(mapf, float, float, V, cols, i, float, M,
	       const long offset = i*cols;
	       float sum       = 0.0;
	       for(long j=0; j < cols; ++j)
		   sum += M[offset + j] * V[j];
	       return sum;
);

/* this is the task used in the OpenCL map 
 * 
 */
struct oclTask: public baseOCLTask<oclTask, float, float> {
    oclTask() {}
    oclTask(size_t cols, size_t rows, float *M, float *V, float *R):
        cols(cols),rows(rows),M(M),V(V),R(R),copy(true) {}
	
    void setTask(const oclTask *t) { 
        assert(t);
        setInPtr(t->V,  t->cols);         // input 
        setOutPtr(t->R, t->rows);         // output
        const long N = t->rows * t->cols;      
        setEnvPtr(t->M, N, t->copy);      // env1. Does the env1 have to be retained ?
    }
    // true if the env (i.e. M) has to be copied to the device
    // false if the previous env has to be used
    void copyEnv(bool k) { copy = k; }

    void setM(float *M2) { M = M2; };
    void setV(float *V2) { V = V2; }

    size_t cols, rows;
    float *M, *V, *R;
    bool  copy;
};

void checkResult(size_t cols, size_t rows, float *M, float *V, float *R) {
#if defined(CHECK)
    bool wrong = false;
    for(size_t i=0;i<rows; ++i) {
        float sum = 0.0;
        for(size_t j=0;j<cols; ++j)
            sum += M[i*cols + j] * V[j];
        if (R[i] != sum) {
            wrong = true;
            std::cerr << "Wrong result " << R[i] << " expected " << sum << "\n";
        }
    }
    if (!wrong) std::cerr << "The result is correct!\n";
#endif
}


int main(int argc, char *argv[]) {
    // default values
    size_t cols = 1000;   
    size_t rows = 10;
    if (argc > 1) {	
        if (argc < 3) {
            std::cerr << "use: " << argv[0] 
                      << " cols rows\n";
            return -1;
        }
        cols = atol(argv[1]);
        rows = atol(argv[2]);	
    }
    
    float *M = new float [rows*cols];
    float *V = new float [cols];   
    float *R = new float [rows];
    assert(M); assert(V);

    for(size_t i=0;i<rows; ++i) {
        for(size_t j=0;j<cols; ++j)
            M[i*cols + j] = (float)(i*cols + j + 1);
    }
    // 1st input
    for(size_t j=0;j<cols; ++j) V[j] = 1.0;

    oclTask oclt(cols, rows, M, V, R);
    ff_mapOCL<oclTask> mv(oclt, mapf);

    mv.run_then_freeze(); mv.wait_freezing();

    checkResult(cols,rows,M,V,R);

    // 2nd input
    for(size_t j=0;j<cols; ++j) V[j] = 2.0; 

    oclt.copyEnv(false);
    oclt.setV(V);
    mv.setTask(oclt);
    mv.run_then_freeze();
    
    // do something else here while the map is running 

    float *M2 = new float[cols*rows];
    for(size_t i=0;i<rows; ++i) {
        for(size_t j=0;j<cols; ++j)
            M2[i*cols + j] = (float)(i + j);
    }

    mv.wait_freezing();
    checkResult(cols,rows,M,V,R);
    delete [] M;

    // now change the environment 
    oclt.copyEnv(true);
    oclt.setM(M2);
    mv.setTask(oclt);
    mv.run_and_wait_end();
    checkResult(cols,rows,M2,V,R);

    return 0;
}
