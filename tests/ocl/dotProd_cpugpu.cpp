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

#define CHECK 1

FF_OCL_STENCIL_ELEMFUNC_ENV(mapf, float, size, i, A, float, B, (void)size; return A[i] * B[i];);
FF_OCL_REDUCE_COMBINATOR(reducef, float, x, y, return (x+y););

struct Task: public baseOCLTask<Task, float, float> {
	Task() :
			result(0.0) {
	}
	Task(const std::vector<float> &A, const std::vector<float> &B) :
			A(A), B(B), M(A.size()), result(0.0) {
	}
	void setTask(Task *t) {
		assert(t);
		setInPtr(const_cast<float*>(t->A.data()), t->A.size());
		setEnvPtr(const_cast<float*>(t->B.data()), t->B.size());
		setOutPtr(const_cast<float*>(t->M.data()), t->A.size());
		setReduceVar(&(t->result));
	}

	void setResult(const float r) {result = r;}

	float combinator(float const &x, float const &y) {return x+y;}

	std::vector<float> A, B;
	std::vector<float> M;
	//float initVal;
	float result;
};

struct Map: ff_Map<Task, Task, float> {

	Task *svc(Task *in) {
		std::vector<float> &A = in->A;
		std::vector<float> &B = in->B;
		const size_t arraySize = A.size();

		float sum = 0.0;
		ff_Map<Task, Task, float>::parallel_reduce(sum, 0.0F, 0, arraySize,
				[&A,&B](const long i, float &sum) {sum += A[i] * B[i];}, [](float &v, const float e) {v += e;});

		in->setResult(sum);
		return in;
	}
};

int main(int argc, char * argv[]) {
	size_t arraySize = 1024;
	int cpugpu = 1;
	if (argc > 3) {
		std::cerr << "use: " << argv[0] << " size 0|1\n";
		std::cerr << "    0: CPU\n";
		std::cerr << "    1: GPU\n";
		return -1;
	}
	if (argc > 1)
		arraySize = atol(argv[1]);
	if (argc > 2)
		cpugpu = atoi(argv[2]);
	assert(cpugpu == 0 || cpugpu == 1);
	int gpudev = clEnvironment::instance()->getCPUDevice(); // this exists for sure
	if (cpugpu != 0) {
		int tmp;
		if ((tmp = clEnvironment::instance()->getGPUDevice()) == -1) {
			std::cerr << "NO GPU device available, switching to CPU\n";
			cpugpu = 0;
		} else
			gpudev = tmp;
	}

	std::vector<float> A(arraySize);
	std::vector<float> B(arraySize);

	for (size_t i = 0; i < arraySize; ++i) {
		A[i] = 1.0;
		B[i] = i;
	}

	Map map;
	ff_mapReduceOCL_1D<Task> oclmap(mapf, reducef, 0.0);

	std::vector<cl_device_id> dev;
	dev.push_back(clEnvironment::instance()->getDevice(gpudev)); //convert logical to OpenCL Id
	oclmap.setDevices(dev);

	Task task(A, B);

	ff_nodeSelector<Task> selector(task);
	selector.addNode(map);
	selector.addNode(oclmap);

	selector.selectNode(cpugpu);
	selector.run_and_wait_end();

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
