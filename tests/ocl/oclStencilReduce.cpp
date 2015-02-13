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
using namespace ff;

#define SIZE 1024
#define NITER 5
#define STREAMLEN 10

FF_OCL_STENCIL_COMBINATOR(reducef, float, x, y, return (x+y););

FF_OCL_STENCIL_ELEMFUNC(mapf, float, float, N, i, in, i_, int, env, char, env2,
//return in[i_] + env[i_] + (i>0) * in[i_-1] + (i<N) * in[i_+1];
		return in[i_] + (i>0) * in[i_-1] + (i<N) * in[i_+1];);

struct oclTask: public baseOCLTask<oclTask, float, float, int, int> {
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
	bool iterCondition(Tout x, unsigned int iter) {
		return iter < NITER;
	}

	float *M_in, *M_out;
	const int size;
	int *env;
};

void init(float *M_in, float *out, int *env, int size) {
	for (int j = 0; j < size; ++j) {
		M_in[j] = (float) j;
		env[j] = j * 10;
		out[j] = 0.0;
	}
}

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
		unsigned int size = t->size;
		printf("[streaming] res[%d]=%.2f\n", size / 2, t->M_out[size / 2]);
		return task;
	}
};

float expected(int idx, int size) {
	float *in = new float[size], *M_out = new float[size];
	int *env = new int[size];
	init(in, M_out, env, size);
	float *tmp = in;
	in = M_out;
	M_out = tmp;
	for (int k = 0; k < NITER; ++k) {
		float *tmp = in;
		in = M_out;
		M_out = tmp;
		for (int i = 0; i < size; ++i)
			M_out[i] = in[i] + (i > 0) * in[i - 1] + (i < size) * in[i + 1];
	}
	float res = M_out[idx];
	delete[] in;
	delete[] M_out;
	delete[] env;
	return res;
}

int main(int argc, char * argv[]) {
	//one-shot
	int size = SIZE;
	if (argc > 1)
		size = atol(argv[1]);
	printf("arraysize = %d\n", size);
	float *M_in = new float[size], *M_out = new float[size];
	int *env = new int[size];
	init(M_in, M_out, env, size);
	oclTask oclt(M_in, M_out, env, size);
	ff_stencilReduceOCL_1D<oclTask> oclStencilReduceOneShot(oclt, mapf, reducef,
			0, 1, 1);
	oclStencilReduceOneShot.run_and_wait_end();
	printf("[oneshot] res[%d]=%.2f\n", size / 2, M_out[size / 2]);
	//printf("reduceVar=%.2f\n", oclt.getReduceVar());

	//stream
	Emitter e(size);
	ff_pipeline pipe;
	pipe.add_stage(&e);
	pipe.add_stage(new ff_stencilReduceOCL_1D<oclTask>(mapf, reducef, 0, 1, 1));
	Printer p;
	pipe.add_stage(&p);
	pipe.run_and_wait_end();

	printf("expected=%.2f\n", expected(size / 2, size));

	return 0;
}
