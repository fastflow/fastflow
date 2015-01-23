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

#include <ff/pipeline.hpp>
#include <ff/stencilReduceOCL.hpp>
using namespace ff;

#define SIZE 1024
#define NITER 5
#define STREAMLEN 10

FFREDUCEFUNC(reducef, float, x, y, return (x+y););

FF_ARRAY_CENV(mapf, float, in, N, i, int, env, return in[i] + env[i];);

struct oclTask: public baseOCLTask<oclTask, float, float, int, int> {
	oclTask() :
			M_in(NULL), M_out(NULL), size(0), env(NULL) {
	}
	oclTask(float *M_in_, float *M_out_, int *env_, size_t size_) :
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
		setSizeOut(t->size);
		setEnvPtr1(t->env);
		setSizeEnv1(t->size);
	}
	bool stop() {
		return getIter() == NITER;
	}

	float *M_in, *M_out;
	const size_t size;
	int *env;
};

class Emitter: public ff_node {
public:
	Emitter(size_t size_) :
			n(0), size(size_) {
		for (int i = 0; i < STREAMLEN; ++i) {
			float *M_in = new float[size];
			int *env = new int[size];
			for (size_t j = 0; j < size; ++j) {
				M_in[j] = j + 1.0;
				env[j] = j * 10;
			}
			float *M_out = new float[size];
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
	size_t n, size;
	oclTask *tasks[STREAMLEN];
};

class Worker: public ff_node
//: public ff_stencilReduceOCL<oclTask>
{
public:
//	Worker() :
//			ff_stencilReduceOCL<oclTask>(mapf, reducef) {
//	}

	virtual void *svc(void *task) {
		oclTask *t = (oclTask *) task;
		//ff_stencilReduceOCL<oclTask>::svc(t);
		for(size_t i=0; i<t->size; ++i) {
			t->M_out[i] = t->M_in[i] + t->env[i];
		}
		printf("arraysize = %ld\n", t->size);
		printf("res[%d]=%.2f\n", t->size / 2, t->M_out[t->size / 2]);
		//printf("reduceVar=%.2f\n", oclt.getReduceVar());
		//delete t;
		return GO_ON;
	}
};

int main(int argc, char * argv[]) {
	//one-shot
	size_t size = SIZE;
	if (argc > 1)
		size = atol(argv[1]);
	printf("arraysize = %ld\n", size);
	float *M_in = new float[size];
	int *env = new int[size];
	for (size_t j = 0; j < size; ++j) {
		M_in[j] = j + 1.0;
		env[j] = j * 10;
	}
	float *M_out = new float[size];
	oclTask oclt(M_in, M_out, env, size);
	ff_stencilReduceOCL<oclTask> oclStencilReduceOneShot(oclt, mapf, reducef);
	oclStencilReduceOneShot.run_and_wait_end();
	printf("res[%d]=%.2f\n", size / 2, M_out[size / 2]);
	//printf("reduceVar=%.2f\n", oclt.getReduceVar());

	//stream
	Emitter e(size);
	ff_pipeline pipe;
	pipe.add_stage(&e);
	//pipe.add_stage(new Worker());
	pipe.add_stage(new ff_stencilReduceOCL<oclTask>(mapf, reducef));
	pipe.run_and_wait_end();

	return 0;
}
