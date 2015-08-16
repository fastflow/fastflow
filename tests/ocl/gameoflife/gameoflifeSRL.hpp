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
 * 'height' is the global number of rows in the input array
 * 'width' is the global number of columns in the input array
 * 'row' is the global row-index
 * 'col' is the global column-index
 * 'env1T' is the element type of the constant environment array
 * 'code' is the OpenCL code of the elemental function
 */
FF_OCL_STENCIL_ELEMFUNC1_2D(mapf,unsigned char,h,w,r,c,unsigned int,
		unsigned char alive_in = GET_IN(r,c);
		unsigned char naliven = 0;
        naliven += r>0 && c>0 ? GET_IN(r-1,c-1) : 0; //top-left
        naliven += r>0 ? GET_IN(r-1,c) : 0; //top-center
        naliven += r>0 && c < w-1 ? GET_IN(r-1,c+1) : 0; //top-right
        naliven += c < w-1 ? GET_IN(r,c+1) : 0; //middle-right
        naliven += r < w-1 && c < w-1 ? GET_IN(r+1,c+1) : 0; //bottom-right
        naliven += r < w-1 ? GET_IN(r+1,c) : 0; //bottom-center
        naliven += r < w-1 && c>0 ? GET_IN(r+1,c-1) : 0; //bottom-left
        naliven += c>0 ? GET_IN(r,c-1) : 0; //middle-left
        return (naliven == 3 || (alive_in && naliven == 2));
		);

FF_OCL_STENCIL_COMBINATOR(reducef, unsigned char, x, y,
		return x||y;
		);

struct oclTaskGol: public ff::baseOCLTask_2D<oclTaskGol, unsigned char> {
	oclTaskGol() :
			M_in(NULL), M_out(NULL), size(0), niters(0), someone(0), nrows_ptr(new unsigned int(0)) {
	}
	oclTaskGol(unsigned char *M_in_, unsigned char *M_out_, unsigned long size_, unsigned int niters_, unsigned int nrows) :
			M_in(M_in_), M_out(M_out_), size(size_), niters(niters_), someone(0), nrows_ptr(new unsigned int(nrows)) {
	}
	~oclTaskGol() {delete nrows_ptr;}

	void setTask(const oclTaskGol *t) {
		assert(t);
		setInPtr(t->M_in, t->size);
		setOutPtr(t->M_out, t->size);
		setEnvPtr(t->nrows_ptr, 1, true);
		setReduceVar(&(t->someone));
		setHeight(*t->nrows_ptr);
		setWidth(*t->nrows_ptr);
		niters = t->niters;
	}
	bool iterCondition(const unsigned char &x, unsigned long iter) {
		return iter < niters;
	}

	unsigned char *M_in, *M_out;
	const unsigned int size;
	unsigned int niters;
	unsigned char someone;
	unsigned int *nrows_ptr;
};



#endif /* GAMEOFLIFESRL_HPP_ */
