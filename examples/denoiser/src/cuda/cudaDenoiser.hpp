/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

/*
 *
 *  Authors:
 *    Maurizio Drocco
 *    Guilherme Peretti Pezzi 
 *  Contributors:  
 *    Marco Aldinucci
 *    Massimo Torquati
 *
 *  First version: February 2014
 */

#ifndef CUDADENOISER_HPP_
#define CUDADENOISER_HPP_

#include <Denoiser.hpp>
//#include <cuda/spd/kernel_cuda/cuda_kernel.cu>
#include <ff/node.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <float.h>

using namespace ff;

extern void kernel_wrapper(unsigned char *d_in, unsigned char *d_out, int *d_noisyMap, unsigned int *noisy, unsigned int n_noisy,
			   unsigned int w, unsigned int h, float alfa, float beta, int nBlocks, int threadsPerBlock );

template<typename DenoiserKernelType>
class cudaDenoiser: public Denoiser, public ff_node {
public:
  cudaDenoiser(void *kernel_params, unsigned int height, unsigned int width, bool fixed_cycles, unsigned int max_cycles, bool trace_time) :
    Denoiser(height, width, fixed_cycles, max_cycles, trace_time)
  {
    d_in = d_out = NULL;
    d_noisy = NULL;
    d_noisyMap = NULL;
  }

  void *svc (void *t) {return Denoiser::svc(t);}

  void svc_end(){
  }

  unsigned int restore(unsigned char *in, unsigned char *out, int *noisymap, unsigned int *noisy, unsigned int n_noisy, void *task) {

    unsigned int height = this->height;
    unsigned int width = this->width;
    bool fixed_cycles = this->fixed_cycles;
    unsigned int max_cycles = this->max_cycles;

    memcpy(out, in, height * width * sizeof(unsigned char));

    // Get number of devices
    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0)
      throw("No CUDA devices, quitting...");

    int threadsPerBlock = 1024;
//    int nBlocks = (height*width +  threadsPerBlock-1)/ threadsPerBlock ; // map
    int nBlocks = (n_noisy +  threadsPerBlock-1)/ threadsPerBlock ; //map orig

    if (cudaSuccess != cudaMalloc((void**) &d_in, height * width * sizeof(unsigned char)))
      throw("cudaMalloc error");
    if (cudaSuccess != cudaMalloc((void**) &d_out, height * width * sizeof(unsigned char)) )
      throw("cudaMalloc error");
    if (cudaSuccess != cudaMalloc((void**) &d_noisyMap, height * width * sizeof(int)) )
      throw("cudaMalloc error");
    if (cudaSuccess != cudaMalloc((void**) &d_noisy, n_noisy * sizeof(unsigned int)) )
      throw("cudaMalloc error");

    if (cudaSuccess != cudaMemcpy(d_noisyMap, noisymap, height * width * sizeof(int), cudaMemcpyHostToDevice) )
      throw("cudaMemcpy error");
    if (cudaSuccess != cudaMemcpy(d_noisy, noisy, n_noisy * sizeof(unsigned int), cudaMemcpyHostToDevice) )
      throw("cudaMemcpy error");

    if (cudaSuccess != cudaMemcpy(d_in, in, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice) )
      throw("cudaMemcpy error");
    if (cudaSuccess != cudaMemcpy(d_out, out, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice) )
      throw("cudaMemcpy error");

    int i;

    unsigned char *tmp;
    tmp = d_out;
    d_out = d_in;
    d_in = tmp;

    for(i=0;i<max_cycles; i++){

    	tmp = d_out;
    	d_out = d_in;
    	d_in = tmp;

      kernel_wrapper(d_in, d_out, d_noisyMap, d_noisy, n_noisy, width, height, 1.3, 5.0, nBlocks, threadsPerBlock);
      std::cout << ".";

    }

    if (cudaSuccess != cudaMemcpy(out, d_out, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost) )
      throw("cudaMemcpy error");

    //clean-up
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_noisyMap);
    cudaFree(d_noisy);

    //free(diff);
    //free(residuals);
    return i; //restore_cycles;
  }

private:
  unsigned char * d_in;
  unsigned char * d_out;
  int * d_noisyMap;
  unsigned int * d_noisy;
};
#endif /* CUDADENOISER_HPP_ */
