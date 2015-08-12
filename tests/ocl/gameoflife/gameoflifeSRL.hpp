/*
 * gameoflifeSRL.hpp
 *
 *  Created on: Jun 2, 2015
 *      Author: droccom
 */

#ifndef GAMEOFLIFESRL_HPP_
#define GAMEOFLIFESRL_HPP_

#define FF_OCL
#include <ff/stencilReduceOCL.hpp>
#include <ff/stencilReduceOCL_macros.hpp>

/* API:
 * 'name' is the name of the string variable in which the code is stored
 * 'inT' is the element type of the input
 * 'size' is the global size of the input array
 * 'idx' is the global index
 * 'in' is the device-local input array
 * 'idx_' is the device-local index
 * 'env1T' is the element type of the constant environment array
 * 'env1' is the constant environment array
 * 'env2T' is the element type of the constant environment value
 * 'env2' is (a pointer to) the constant environment value
 * 'code' is the OpenCL code of the elemental function
 */
FF_OCL_STENCIL_ELEMFUNC1(mapf,unsigned char,length,idx,in,idx_,unsigned long,env1,
		unsigned long w = *env1;
		unsigned long r = idx / w;
		unsigned long c = idx % w;
		unsigned char alive_in = in[idx_];
		unsigned char naliven = 0;
        unsigned long rr = (r>0)?1:0;
        unsigned long cc = (c>0)?1:0;
        unsigned long cw = (c < w-1)?1:0;
        unsigned long rw = (r < w-1)?1:0;
		naliven += rr*cc*in[idx_ - w - 1];
		naliven += rr*in[idx_ - w]; //top-center
		naliven += r>0 && c < w-1 ? in[idx_ - w + 1] : 0; //top-right
		naliven += cw*in[idx_ + 1]; //middle-right
		naliven += rw*cw*in[idx_ + w + 1]; //bottom-right
		naliven += rw*in[idx_ + w]; //bottom-center
		naliven += rw*cc*in[idx_ + w - 1]; //bottom-left
		naliven += cc*in[idx_ - 1]; //middle-left
		return (naliven == 3 || (alive_in && naliven == 2));
		);

FF_OCL_STENCIL_COMBINATOR(reducef, unsigned char, x, y,
		return x||y;
		);

struct oclTask: public ff::baseOCLTask<oclTask, unsigned char> {
	oclTask() :
			M_in(NULL), M_out(NULL), size(0), niters(0), someone(0), nrows_ptr(new unsigned long(0)) {
	}
	oclTask(unsigned char *M_in_, unsigned char *M_out_, unsigned long size_, unsigned int niters_, unsigned long nrows) :
			M_in(M_in_), M_out(M_out_), size(size_), niters(niters_), someone(0), nrows_ptr(new unsigned long(nrows)) {
	}
	~oclTask() {delete nrows_ptr;}

	void setTask(const oclTask *t) {
		assert(t);
		setInPtr(t->M_in, t->size);
		setOutPtr(t->M_out, t->size);
		setEnvPtr(t->nrows_ptr, 1, true);
		setReduceVar(&(t->someone));
		niters = t->niters;
	}
	bool iterCondition(const unsigned char &x, unsigned long iter) {
		return iter < niters;
	}

	unsigned char *M_in, *M_out;
	const unsigned long size;
	unsigned long niters;
	unsigned char someone;
	unsigned long *nrows_ptr;
};



#endif /* GAMEOFLIFESRL_HPP_ */
