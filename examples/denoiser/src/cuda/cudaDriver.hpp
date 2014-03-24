/*
 * cudaDriver.hpp
 *
 *  Created on: Feb 24, 2014
 *      Author: droccom
 */

#ifndef CUDADRIVER_HPP_
#define CUDADRIVER_HPP_

#include <Driver.hpp>
#include <cuda/cudaDenoiser.hpp>

template<typename DetectorKernelType, typename DenoiserKernelType>
class cudaDriver: public Driver<DetectorKernelType> {
public:
	//using Driver<DetectorKernelType>::Driver; not supported in gcc < 4.8
	cudaDriver(arguments args, bool trace_time = false) : Driver<DetectorKernelType>(args, trace_time) {}

	ff_node *createDenoiserNode(parallel_parameters_t *pp,
			void *denoiser_kernel_params) {
		return new cudaDenoiser<DenoiserKernelType>(denoiser_kernel_params,
				this->height, this->width, this->fixed_cycles, this->max_cycles,
				this->trace_time);
	}

	unsigned int get_worker_cycles(ff_node *w) {
		return ((cudaDenoiser<DenoiserKernelType> *) w)->getCycles();
	}

	double get_worker_svc(ff_node *w) {
		return ((cudaDenoiser<DenoiserKernelType> *) w)->getSvcTime();
	}

	void displayHeadersParallel(parallel_parameters_t *pp) {
		cout << "* cuda parallel configuration:" << endl << "n. farm workers: "
				<< this->n_farm_workers << endl << "n. detector workers: "
				<< this->n_detector_kernel_workers << endl << "*" << endl;
	}
};

#endif /* CUDADRIVER_HPP_ */
