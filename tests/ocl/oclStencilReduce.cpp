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
/* Author: Maurizio Drocco
 *         
 */

#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif

#include <ff/pipeline.hpp>
#include <ff/stencilReduceOCL.hpp>
#include <sstream>
#include <cstdio>
#include <limits>
using namespace ff;

#define SIZE 1024
#define NITERS 1
#define STREAMLEN 10
#define NACCELERATORS 2
#define WINWIDTH 5

size_t niters;
typedef int basictype;

FF_OCL_STENCIL_COMBINATOR(reducef, int, x, y, return (x+y) );

FF_OCL_STENCIL_ELEMFUNC1(mapf, int, N, i, in, i_, int, env,
		int res = in[i_];
	//if(i>0) res += in[i_-1]; if(i<(N-1)) res += in[i_+1];
	++res;
	return res;
);

void init(basictype *M_in, basictype *out, int *env, int size) {
	for (int j = 0; j < size; ++j) {
		M_in[j] = j;
		env[j] = j * 10;
		out[j] = 0;
	}
}

basictype mapf_(basictype *in, int i, int *env, char *env2) {
	basictype res = in[i];
	++res;
	return res;
}

basictype reducef_(basictype x, basictype y) {
	return x+y;
}

void print_res(const char label[], basictype *M, basictype r, int size) {
	for (int i = 0; i < size; ++i)
		std::cout << "[" <<label<< " ] res["<<i<<"]="<<M[i]<<"\n";
	std::cout << "[" <<label<< " ] reduceVar = "<<r<<"\n";
}

void print_expected(int size) {
	basictype *in = new basictype[size], *M_out = new basictype[size];
	int *env = new int[size];
	init(in, M_out, env, size);
	basictype *tmp = in;
	in = M_out;
	M_out = tmp;
	basictype red = 0;
	for (unsigned int k = 0; k < niters; ++k) {
		basictype *tmp = in;
		in = M_out;
		M_out = tmp;
		for (int i = 0; i < size; ++i)
			M_out[i] = mapf_(in, i, env, NULL);
		//reduce
		red = 0;
		for (int i = 0; i < size; ++i)
			red = reducef_(red, M_out[i]);
		std::stringstream l;
		l << "EXPECTED_" << k;
		//print_res(l.str().c_str(), M_out, size);
	}
	print_res("EXPECTED", M_out, red, size);
	delete[] in;
	delete[] M_out;
	delete[] env;
}

void check(basictype *M, basictype r, int size) {
	unsigned int ndiff = 0;
	basictype *in = new basictype[size], *M_out = new basictype[size];
	int *env = new int[size];
	init(in, M_out, env, size);
	basictype *tmp = in;
	in = M_out;
	M_out = tmp;
	basictype red = 0;
	for (unsigned int k = 0; k < niters; ++k) {
		basictype *tmp = in;
		in = M_out;
		M_out = tmp;
		for (int i = 0; i < size; ++i)
			M_out[i] = mapf_(in,i,env,NULL);
		//reduce
		red = 0;
		for (int i = 0; i < size; ++i)
			red = reducef_(red, M_out[i]);
	}
	for (int i = 0; i < size; ++i)
		if (M[i] != M_out[i]) {
			ndiff++;
			std::cout<<"out["<<i<<"]="<<M[i]<<", check["<<i<<"]="<<M_out[i]<<"\n";
		}
	if (red != r) {
		ndiff++;
		std::cout<<"REDUCE-computed="<<r<<", REDUCE-expected="<<red<<"\n";
	}
		std::cout<<"check summary: "<<ndiff<<" diffs\n";
		//std::cout <<"expected REDUCE = " << red << " (max " << std::numeric_limits<basictype>::max() << ")\n";
}

struct oclTask: public baseOCLTask<oclTask, basictype, int> {
	oclTask() :
			M_in(NULL), M_out(NULL), result(0), size(0), env(NULL) {
	}
	oclTask(basictype *M_in_, basictype *M_out_, int *env_, int size_) :
			M_in(M_in_), M_out(M_out_), result(0), size(size_), env(env_) {
	}
	~oclTask() {
		delete[] M_in;
		delete[] M_out;
		delete[] env;
	}
	void setTask(const oclTask *t) {
		assert(t);
		setInPtr(t->M_in, t->size);
		setOutPtr(t->M_out);
		setEnvPtr(t->env, t->size);
		setReduceVar(&(t->result));
	}
	bool iterCondition(const Tin &x, size_t iter) {
		return iter < niters;
	}

	virtual basictype combinator(basictype x, basictype y) {
		return x + y;
	}

	basictype *M_in, *M_out, result;
	const int size;
	int *env;
};

class Emitter: public ff_node {
public:
	Emitter(int size_) :
			n(0), size(size_) {
		for (int i = 0; i < STREAMLEN; ++i) {
			basictype *M_in = new basictype[size], *M_out = new basictype[size];
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
    if (argc > 1) {
        if (argc == 2) {
            //printf("-> %d %s\n",argc,argv[1]);
            if (argv[1] != std::string("-h")) 
                size = atol(argv[1]);
            else {
                std::cout << "Usage: arraysize accelerators niters (ex. "
                          << size << " " << nacc << " " << niters << ")" << std::endl;
                return 0;
            }
        }
        if (argc > 2)
                 nacc = atol(argv[2]);
        if (argc > 3)
            niters = atol(argv[3]);
    }        
	basictype *M_in = new basictype[size], *M_out = new basictype[size];
	int *env = new int[size];
	init(M_in, M_out, env, size);
	oclTask oclt(M_in, M_out, env, size);
	ff_stencilReduceLoopOCL_1D<oclTask> oclStencilReduceOneShot(oclt, mapf,	reducef, 0, nacc, WINWIDTH);
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
	check(M_out, oclt.result, size);

	return 0;
}
