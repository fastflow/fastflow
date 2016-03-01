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

#define CHECK 1
#ifdef CHECK
#include "ctest.h"
#else
#define NACC 1
#endif

#define SIZE 1024
#define NITERS 100
#define WINWIDTH 5

size_t niters;
typedef int basictype;

FF_OCL_REDUCE_COMBINATOR(reducef, int, x, y, return (x+y));

FF_OCL_STENCIL_ELEMFUNC(mapf, int, useless, i, in,
		(void)useless;
		return in[i] + 1;
);

//implicit input
//FF_OCL_STENCIL_ELEMFUNC_1D(mapf, int, N, i,
//		(void)N;
//		return GET_IN(i) + 1;
//);

void init(basictype *M_in, basictype *out, int size) {
	for (int j = 0; j < size; ++j) {
		M_in[j] = j;
		out[j] = 0;
	}
}

basictype mapf_(basictype *in, int i) {
	basictype res = in[i];
	++res;
	return res;
}

basictype reducef_(basictype x, basictype y) {
	return x + y;
}

void print_res(const char label[], basictype *M, basictype r, int size) {
	for (int i = 0; i < size; ++i)
		std::cout << "[" << label << " ] res[" << i << "]=" << M[i] << "\n";
	std::cout << "[" << label << " ] reduceVar = " << r << "\n";
}

unsigned int check(basictype *M, basictype r, int size) {
	unsigned int ndiff = 0;
	basictype *in = new basictype[size], *M_out = new basictype[size];
	init(in, M_out, size);
	basictype *tmp = in;
	in = M_out;
	M_out = tmp;
	basictype red = 0;
	for (unsigned int k = 0; k < niters; ++k) {
		basictype *tmp = in;
		in = M_out;
		M_out = tmp;
		for (int i = 0; i < size; ++i)
			M_out[i] = mapf_(in, i);
		//reduce
		red = 0;
		for (int i = 0; i < size; ++i)
			red = reducef_(red, M_out[i]);
	}
	for (int i = 0; i < size; ++i)
		if (M[i] != M_out[i]) {
			ndiff++;
			std::cout << "out[" << i << "]=" << M[i] << ", check[" << i << "]="
					<< M_out[i] << "\n";
		}
	if (red != r) {
		ndiff++;
		std::cout << "REDUCE-computed=" << r << ", REDUCE-expected=" << red
				<< "\n";
	}
	std::cout << "check summary: " << ndiff << " diffs\n";
	//std::cout <<"expected REDUCE = " << red << " (max " << std::numeric_limits<basictype>::max() << ")\n";
	delete[] in;
	delete[] M_out;
	return ndiff;
}

struct oclTask: public baseOCLTask<oclTask, basictype> {
	oclTask() :
			M_in(NULL), M_out(NULL), result(0), size(0) {
	}
	oclTask(basictype *M_in_, basictype *M_out_, int size_) :
			M_in(M_in_), M_out(M_out_), result(0), size(size_) {
	}
	void setTask(oclTask *t) {
		assert(t);
		setInPtr(t->M_in, t->size);
		setOutPtr(t->M_out, t->size);
		setReduceVar(&(t->result));
	}
	bool iterCondition(const Tin &x, size_t iter) {
		return iter < niters;
	}

	virtual basictype combinator(basictype const &x, basictype const &y) {
		return x + y;
	}

	basictype *M_in, *M_out, result;
	const int size;
};

int main(int argc, char * argv[]) {
	//one-shot
	int size = SIZE;
	int nacc = NACC;
	niters = NITERS;
	if (argc > 1) {
		if (argc == 2) {
			//printf("-> %d %s\n",argc,argv[1]);
			if (argv[1] != std::string("-h"))
				size = atol(argv[1]);
			else {
				std::cout << "Usage: arraysize accelerators niters (ex. "
						<< size << " " << nacc << " " << niters << ")"
						<< std::endl;
				return 0;
			}
		}
		if (argc > 2)
			nacc = atol(argv[2]);
		if (argc > 3)
			niters = atol(argv[3]);
	}
	basictype *M_in = new basictype[size], *M_out = new basictype[size];
	init(M_in, M_out, size);
	oclTask oclt(M_in, M_out, size);
	ff_stencilReduceLoopOCL_1D<oclTask> oclStencilReduceOneShot(oclt, mapf,
			reducef, 0, nullptr, nacc, WINWIDTH);
	SET_DEVICE_TYPE(oclStencilReduceOneShot);
	oclStencilReduceOneShot.run_and_wait_end();


#ifdef CHECK
	if (check(M_out, oclt.result, size)) {
		printf("Error\n");
		exit(1); //ctest
	}
#endif
	delete[] M_in;
	delete[] M_out;
	return 0;
}
