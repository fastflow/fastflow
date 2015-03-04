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
 *         torquati@di.unipi.it  massimotor@gmail.com
 */

#define FF_OCL

#include <ff/pipeline.hpp>
#include <ff/stencilReduceOCL.hpp>
#include <sstream>
#include <cstdio>
using namespace ff;

#define SIZE 1024
#define NITERS 1
#define STREAMLEN 10
#define NACCELERATORS 2
#define WINWIDTH 5

unsigned int niters;

FF_OCL_STENCIL_COMBINATOR(reducef, float, x, y, return (x+y);)
;

FF_OCL_STENCIL_ELEMFUNC(mapf, float, N, i, in, i_, int, env, char, env2,
		float res = in[i_];
	if(i>0) res += in[i_-1]; if(i<(N-1)) res += in[i_+1];
	return res;
);

void init(float *M_in, float *out, int *env, int size) {
	for (int j = 0; j < size; ++j) {
		//M_in[j] = (float) j;
		M_in[j] = 1;
		env[j] = j * 10;
		out[j] = 0.0;
	}
}

void print_res(const char label[], float *M, float r, int size) {
	for (int i = 0; i < size; ++i)
		printf("[%s] res[%d]=%.2f\n", label, i, M[i]);
	printf("[%s] reduceVar = %.2f\n", label, r);
}

void print_expected(int size) {
	float *in = new float[size], *M_out = new float[size];
	int *env = new int[size];
	init(in, M_out, env, size);
	float *tmp = in;
	in = M_out;
	M_out = tmp;
	float red = 0;
	for (unsigned int k = 0; k < niters; ++k) {
		float *tmp = in;
		in = M_out;
		M_out = tmp;
		for (int i = 0; i < size; ++i)
			M_out[i] = in[i]
		+ (i > 0) * in[i - 1] + (i < size - 1) * in[i + 1];
		//reduce
		red = 0;
		for (int i = 0; i < size; ++i)
			red += M_out[i];
		std::stringstream l;
		l << "EXPECTED_" << k;
		//print_res(l.str().c_str(), M_out, size);
	}
	print_res("EXPECTED", M_out, red, size);
	delete[] in;
	delete[] M_out;
	delete[] env;
}

void check(float *M, float r, int size) {
	unsigned int ndiff = 0;
	float *in = new float[size], *M_out = new float[size];
	int *env = new int[size];
	init(in, M_out, env, size);
	float *tmp = in;
	in = M_out;
	M_out = tmp;
	float red = 0;
	for (unsigned int k = 0; k < niters; ++k) {
		float *tmp = in;
		in = M_out;
		M_out = tmp;
		for (int i = 0; i < size; ++i) {
			float res = in[i];
			if (i > 0)
				res += in[i - 1];
			if (i < (size - 1))
				res += in[i + 1];
			M_out[i] = res;
		}
		//reduce
		red = 0;
		for (int i = 0; i < size; ++i)
			red += M_out[i];
	}
	for (int i = 0; i < size; ++i)
		if (M[i] != M_out[i]) {
			ndiff++;
			printf("[i] out[%d]=%.2f, check[%d]=%.2f\n", i, M[i], i, M_out[i]);
		}
	if (red != r) {
		ndiff++;
		printf("[i] REDUCE-computed=%.2f, REDUCE-expected=%.2f\n", r, red);
	}
		printf("check summary: %d diffs\n", ndiff);
}

struct oclTask: public baseOCLTask<oclTask, float, int, int> {
	oclTask() :
			M_in(NULL), M_out(NULL), size(0), env(NULL) {
	}
	oclTask(float *M_in_, float *M_out_, int *env_, int size_) :
			M_in(M_in_), M_out(M_out_), size(size_), env(env_) {
	}
	~oclTask() {
		delete[] M_in;
		delete[] M_out;
		delete[] env;
	}
	void setTask(const oclTask *t) {
		assert(t);
		setInPtr(t->M_in);
		setSizeIn(t->size);
		setOutPtr(t->M_out);
		setEnvPtr1(t->env);
	}
	bool iterCondition(Tin x, unsigned int iter) {
		return iter < niters;
	}

	virtual float combinator(float x, float y) {
		return x + y;
	}

	float *M_in, *M_out;
	const int size;
	int *env;
};

class Emitter: public ff_node {
public:
	Emitter(int size_) :
			n(0), size(size_) {
		for (int i = 0; i < STREAMLEN; ++i) {
			float *M_in = new float[size], *M_out = new float[size];
			int *env = new int[size];
			init(M_in, M_out, env, size);
			tasks[i] = new oclTask(M_in, M_out, env, size);
		}
	}

	~Emitter() {
		for (int i = 0; i < STREAMLEN; ++i) {
			delete tasks[i];
		}
	}

	virtual void *svc(void *task) {
		if (n < STREAMLEN)
			return tasks[n++];
		return EOS;
	}

private:
	int n, size;
	oclTask *tasks[STREAMLEN];
};

class Printer: public ff_node {
public:
	void *svc(void *task) {
		oclTask *t = (oclTask *) task;
		int size = t->size;
		print_res("streaming", t->M_out, /*t->reduceVar*/0, size);
		return task;
	}
};

int main(int argc, char * argv[]) {
	//one-shot
	int size = SIZE;
	int nacc = NACCELERATORS;
	niters = NITERS;
	if (argc > 1)
		size = atol(argv[1]);
	if (argc > 2)
		nacc = atol(argv[2]);
	if (argc > 3)
		niters = atol(argv[3]);
	fprintf(stderr, "arraysize = %d, n. accelerators = %d\n", size, nacc);
	float *M_in = new float[size], *M_out = new float[size];
	int *env = new int[size];
	init(M_in, M_out, env, size);
	oclTask oclt(M_in, M_out, env, size);
	ff_stencilReduceLoopOCL_1D<oclTask> oclStencilReduceOneShot(oclt, mapf,
			reducef, 0, nacc, WINWIDTH);
	oclStencilReduceOneShot.run_and_wait_end();
	//print_res("INPUT", M_in, size);
	//print_res("oneshot", M_out, oclStencilReduceOneShot.getReduceVar(), size);
	//printf("reduceVar=%.2f\n", oclt.getReduceVar());

	//stream
	//	Emitter e(size);
	//	ff_pipeline pipe;
	//	pipe.add_stage(&e);
	//	pipe.add_stage(new ff_stencilReduceOCL_1D<oclTask>(mapf, reducef, 0, nacc, 1));
	//	Printer p;
	//	pipe.add_stage(&p);
	//	pipe.run_and_wait_end();

	//print_expected(size);
	check(M_out, oclStencilReduceOneShot.getReduceVar(), size);

	return 0;
}
