/*
 * cudaDenoiserReduceF.hpp
 *
 *  Created on: Mar 6, 2014
 *      Author: droccom
 */
#ifndef CUDADENOISERREDUCEF_HPP_
#define CUDADENOISERREDUCEF_HPP_

#include <ff/stencilReduceCUDA.hpp>

using namespace ff;


FFREDUCEFUNC(reduceF, float, x, y, return x+y;);


#endif /* CUDADENOISERREDUCEF_HPP_ */
