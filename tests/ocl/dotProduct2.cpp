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
#include <ff/stencilReduceOCL.hpp>
using namespace ff;

#define CHECK 1

FF_OCL_STENCIL_ELEMFUNC_ENV(mapf, float, size, i, A, float, B, { 
                            (void)size; 
                            return A[i] * B[i];
    });
FF_OCL_REDUCE_COMBINATOR(reducef, float, x, y, {
        return (x+y);
    });

struct Task: public baseOCLTask<Task, float, float> {
	Task(): result(0.0) { }
	Task(const std::vector<float> &A, const std::vector<float> &B) :
			A(A), B(B), M(A.size()), result(0.0) {}

	void setTask(Task *t) {
        MemoryFlags flags = {CopyFlags::COPY,ReuseFlags::DONTREUSE,ReleaseFlags::RELEASE};
		setInPtr(const_cast<float*>(t->A.data()), t->A.size(), flags);
		setEnvPtr(const_cast<float*>(t->B.data()), t->B.size(), flags);
		setOutPtr(const_cast<float*>(t->M.data()), t->M.size(), flags);
		setReduceVar(&(t->result));
	}

	void setResult(const float r) {result = r;}

	float combinator(float const &x, float const &y) {return x+y;}

	std::vector<float> A, B, M;
	float result;
};

int main(int argc, char * argv[]) {
	size_t arraySize = 1024;
	if (argc > 2) {
		std::cerr << "use: " << argv[0] << " size\n";
		return -1;
	}
	if (argc > 1) arraySize = atol(argv[1]);

	std::vector<float> A(arraySize), B(arraySize);

	for (size_t i = 0; i < arraySize; ++i) {
		A[i] = 1.0;
		B[i] = i;
	}
    
	std::vector<cl_device_id> dev;
    int tmp = clEnvironment::instance()->getGPUDevice();
    // if a gpu device does not exist, switching to CPU
    if (tmp == -1) {
        std::cout << "selected CPU device\n";
        tmp = clEnvironment::instance()->getCPUDevice();
    } else {
        std::cout << "selected GPU device\n";
    }
	dev.push_back(clEnvironment::instance()->getDevice(tmp)); //convert logical to OpenCL Id

    Task task(A, B);
	ff_mapReduceOCL_1D<Task> oclmap(task, mapf, reducef, 0.0);
	oclmap.setDevices(dev);
    if (oclmap.run_and_wait_end()<0) {
        error("running oclmap\n");
        return -1;
    }

	std::cout << "Result = " << std::setprecision(10) << task.result << "\n";

#if defined(CHECK)
	float r = 0.0;
	for (size_t j = 0; j < arraySize; ++j) {
		r += A[j] * B[j];
	}
	if (r != task.result) {
		printf("Wrong result, should be %.2f\n", r);
		exit(1); //ctest
	} else
		printf("OK\n");
#endif

	return 0;
}
