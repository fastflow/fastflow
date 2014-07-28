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

#ifndef DENOISERCUDATASKGAUSSIAN_HPP_
#define DENOISERCUDATASKGAUSSIAN_HPP_

#include <ff/stencilReduceCUDA.hpp>
using namespace ff;

struct DenoiserKernelGaussian_params {
  float alfa, beta;
};

class DenoiserCUDATaskGaussian: public baseCUDATask<unsigned int, float, unsigned char, unsigned char, int,
		float, unsigned int, float> {
public:
	void setTask(void* t) {
		if (t) {
			denoise_task * task_ = (denoise_task *) t;
			fixed_cycles = task_->fixed_cycles;
			in = task_->input;
			out = task_->output;
			noisy = task_->noisy;
			n_noisy = task_->n_noisy;
			height = task_->height;
			width = task_->width;
			residuals = (float *)malloc(n_noisy * sizeof(float));
			this->setInPtr(noisy);
			this->setSizeIn(n_noisy);
			this->setOutPtr(residuals);
			this->setEnv1Ptr(in);
			this->setSizeEnv1(height * width);
			this->setEnv2Ptr(out);
			this->setSizeEnv2(height * width);
			this->setEnv3Ptr(task_->noisymap);
			this->setSizeEnv3(height * width);
			DenoiserKernelGaussian_params *kparams = (DenoiserKernelGaussian_params *)task_->kernel_params;
			params[0] = kparams->alfa;
			params[1] = kparams->beta;
			this->setEnv4Ptr(params);
			this->setSizeEnv4(2);
			sizes[0] = height;
			sizes[1] = width;
			this->setEnv5Ptr(sizes);
			this->setSizeEnv5(2);
			diff = (float *)malloc(n_noisy * sizeof(float));
			memset(diff, 0, n_noisy * sizeof(float));
			this->setEnv6Ptr(diff);
			this->setSizeEnv6(n_noisy);
		}
	}

  void startMR() {
    setReduceVar(0.0f);
  }

	void beforeMR() {
	  old_reduceVar = getReduceVar();
	}

#define EPS 1e-01f
	bool iterCondition(float reduceVar, size_t iter) {
	  return fixed_cycles || (std::abs)(getReduceVar() - old_reduceVar) / old_reduceVar >= EPS;
	}

  void endMR(void *) {
		cudaMemcpy(out, getEnv2DevicePtr(), height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		free(diff);
	}

	void swap() {
		unsigned char *tmp = getEnv1DevicePtr();
		setEnv1DevicePtr(getEnv2DevicePtr());
		setEnv2DevicePtr(tmp);
	}

private:
	float params[2]; //alfa, beta
	unsigned int sizes[2]; //width, height
	unsigned int *noisy, n_noisy, height, width;
	unsigned char *in, *out;
	float *diff, *residuals;
	float old_reduceVar;
	bool fixed_cycles;
};

#endif /* DENOISERCUDATASKGAUSSIAN_HPP_ */
