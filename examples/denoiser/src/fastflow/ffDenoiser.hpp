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

#ifndef FFDENOISER_HPP_
#define FFDENOISER_HPP_

#include <Denoiser.hpp>

#include <ff/node.hpp>
//#include <ff/parallel_for.hpp>

using namespace ff;

template<typename DenoiserKernelType>
class ffDenoiser: public Denoiser, public ff_node {
public:
  ffDenoiser(unsigned int n_kernel_workers_, void *kernel_params, unsigned int height, unsigned int width, bool fixed_cycles, unsigned int max_cycles, bool trace_time) :
    Denoiser(height, width, fixed_cycles, max_cycles, trace_time),
    n_kernel_workers(n_kernel_workers_) {
    //FF_PARFOR_ASSIGN(pf_den,n_kernel_workers);
    //FF_PARFORREDUCE_ASSIGN(pfr_den, float, n_kernel_workers);
    kernel = new DenoiserKernelType(kernel_params, height, width);
  }

  void *svc (void *task) {return Denoiser::svc(task);}

  void svc_end(){
    //FF_PARFOR_DONE(pf_den);
    //FF_PARFORREDUCE_DONE(pfr_den);
      }

  unsigned int restore(unsigned char *in, unsigned char *out, int *noisymap, unsigned int *noisy, unsigned int n_noisy, void *task) {
      unsigned int height = this->height;
      unsigned int width = this->width;
      bool fixed_cycles = this->fixed_cycles;
      unsigned int max_cycles = this->max_cycles;

      DenoiserKernelType *kernel = this->kernel;
      unsigned char *diff = (unsigned char *) malloc(n_noisy * sizeof(unsigned char));
      memset(diff, 0, n_noisy * sizeof(unsigned char));
      memcpy(out, in, height * width * sizeof(unsigned char));
      float *residuals = (float *) malloc(n_noisy * sizeof(float));
      bool fixed = false;
      unsigned int restore_cycles = 0;
      float old_residual = 0.0f;
      while (true) {
        //(MAP) parallel restore
        unsigned int block_size = std::max((n_noisy + n_kernel_workers - 1) / n_kernel_workers, 1u);
        unsigned int n_blocks = std::max((n_noisy + block_size - 1) / block_size, 1u);
        int b=0;
        //FF_PARFOR_START(pf_den, b, 0, n_blocks, 1, 1, n_kernel_workers)
    for(int b=0; b<n_blocks; ++b)
        {
          unsigned int first = b * block_size;
          unsigned int last = (std::min)(first + block_size - 1, n_noisy - 1);
          kernel->restore_chunk(noisy, first, last, noisymap, in, out);
          for (int i=first ;i<=last ;++i)
          {
            unsigned int x = noisy[i];
            unsigned char newdiff = (unsigned char) (_ABS((int ) (out[x]) - noisymap[x]));
            residuals[i] = (float) (_ABS((int ) newdiff - (int ) (diff[i])));
            diff[i] = newdiff;
          }
        }
    //FF_PARFOR_STOP(pf_den);

        //reduce residuals
        float residual = 0.0f;
        int chunk = (std::max)(1u, (n_noisy + n_kernel_workers - 1) / n_kernel_workers);
        //FF_PARFORREDUCE_START(pfr_den, residual, 0.0f /*identity*/, i, 0, n_noisy, 1, chunk, n_kernel_workers)
	for (unsigned int i = 0; i < n_noisy; ++i)
          residual += residuals[i];
	  //FF_PARFORREDUCE_STOP(pfr_den, residual, +);
        residual /= n_noisy;
        float delta = _ABS(residual - old_residual) / old_residual;
        ++restore_cycles;
        //check convergence
        if (fixed_cycles)
          fixed = restore_cycles == max_cycles;
        else
          fixed = delta < RESIDUAL_THRESHOLD || restore_cycles >= max_cycles;
        if (fixed)
          break;
        old_residual = residual;
        memcpy(in, out, height * width * sizeof(unsigned char));
      }
      //clean-up
      free(diff);
      free(residuals);
      return restore_cycles;
    }

  virtual ~ffDenoiser() {
      if (kernel)
        delete kernel;
    }

private:
  unsigned int n_kernel_workers;
  // FF_PARFOR_DECL(pf_den);
  // FF_PARFORREDUCE_DECL(pfr_den, float);
  DenoiserKernelType *kernel;
};
#endif /* FFDENOISER_HPP_ */
