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
 *        
 */

#if !defined(FF_CUDA)
#define FF_CUDA
#endif

#include <ff/stencilReduceCUDA.hpp>
using namespace ff;


struct mypair { double a; double b; };

FFMAPFUNC2(mapF, double, mypair, p,
	   return p.a*p.b;
);

FFREDUCEFUNC(reduceF, double, result, x,
	     return result+x;
);

struct cudaTask: public baseCUDATask<mypair,double> {

    cudaTask(size_t size,mypair *AB, double *C):size(size),AB(AB),C(C), result(0.0f) {}
    cudaTask():size(0),AB(NULL),C(C), result(0.0f) {}
  
    size_t size;
    mypair *AB;
    double  *C;
    double result;
    
    void setTask(void *t) {
        const cudaTask &task = *(cudaTask*)t;
        
        setInPtr(task.AB);
        setOutPtr(task.C);
        setSizeIn(task.size);
    }
    
    void afterMR(void *t) {
        cudaTask &task = *(cudaTask*)t;
        task.result  = getReduceVar();
    }
};


int main(int argc, char * argv[]) {
  size_t inputsize = 1024;

  if (argc > 1) inputsize = atoi(argv[1]);
  	printf("using arraysize = %lu\n", inputsize);

  mypair *AB = new mypair[inputsize];
  for(long j=0;j<inputsize;++j) {
      AB[j].a=j*3.14; 
      AB[j].b=2.1*j;
  }
  double *C = new double[inputsize];

  cudaTask ct(inputsize, AB, C);
  FFSTENCILREDUCECUDA(cudaTask, mapF, reduceF) dotprod(ct);
  dotprod.run_and_wait_end();

#ifdef CHECK
  double expected = 0;
  for(long j=0;j<inputsize;++j)
	  expected += (AB[j].a * AB[j].b);
  if(expected != ct.result) {
	  std::cerr << "computed="<<ct.result<<" expected="<<expected<<"\n";
	  printf("ERROR\n");
      return 1;
  }
#endif

  printf("Result = %g\n", ct.result);
  return 0;
}

