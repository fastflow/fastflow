/*
 * ffDenoiser.hpp
 *
 *  Created on: Feb 24, 2014
 *      Author: droccom
 */

#ifndef CUDADENOISERAUTO_HPP_
#define CUDADENOISERAUTO_HPP_

#include <Denoiser.hpp>
#include <ff/stencilReduceCUDA.hpp>
#include <cuda/spd/DenoiserMapFSPD.hpp>
#include <cuda/cudaDenoiserReduceF.hpp>
using namespace ff;

template<typename DenoiserCUDAtaskType, typename DenoiserCUDAmapF>
class cudaDenoiserAUTO: public Denoiser, public FFSTENCILREDUCECUDA(DenoiserCUDAtaskType, DenoiserCUDAmapF, reduceF) {
public:
  cudaDenoiserAUTO(void *kernel_params_, unsigned int height, unsigned int width, bool fixed_cycles, unsigned int max_cycles, bool trace_time) :
kernel_params(kernel_params_),
	FFSTENCILREDUCECUDA(DenoiserCUDAtaskType, DenoiserCUDAmapF, reduceF)(max_cycles),
	  Denoiser(height, width, fixed_cycles, max_cycles, trace_time) {
    //cuda_map assign?
  }

  void *svc(void *t) {
	((denoise_task *)t)->kernel_params = kernel_params;
	((denoise_task *)t)->fixed_cycles = fixed_cycles;
    return Denoiser::svc(t);
  }

  void svc_end() {}

  unsigned int restore(unsigned char *in, unsigned char *out, int *noisymap, unsigned int *noisy, unsigned int n_noisy, void *task) {
    unsigned int height = this->height;
    unsigned int width = this->width;
    memcpy(out, in, height * width * sizeof(unsigned char));
    FFSTENCILREDUCECUDA(DenoiserCUDAtaskType, DenoiserCUDAmapF, reduceF)::svc(task);
    return this->getIter();
  }

private:
  void *kernel_params;
};
#endif /* CUDADENOISERAUTO_HPP_ */
