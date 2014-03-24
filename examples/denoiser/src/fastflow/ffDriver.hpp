/*
 * ffDriver.hpp
 *
 *  Created on: Feb 24, 2014
 *      Author: droccom
 */

#ifndef FFDRIVER_HPP_
#define FFDRIVER_HPP_

#include <Driver.hpp>
#include <fastflow/ffDenoiser.hpp>

template<typename DetectorKernelType, typename DenoiserKernelType>
class ffDriver: public Driver<DetectorKernelType> {
public:
	//using Driver<DetectorKernelType>::Driver; not supported in gcc < 4.8
	ffDriver(arguments args, bool trace_time = false) :
			Driver<DetectorKernelType>(args, trace_time), n_denoiser_workers(0) {
	}

	ff_node *createDenoiserNode(parallel_parameters_t *pp,
			void *denoiser_kernel_params) {
		return new ffDenoiser<DenoiserKernelType>(pp->n_denoiser_workers,
				denoiser_kernel_params, this->height, this->width,
				this->fixed_cycles, this->max_cycles, this->trace_time);
	}

	unsigned int get_worker_cycles(ff_node *w) {
		return ((ffDenoiser<DenoiserKernelType> *) w)->getCycles();
	}

	double get_worker_svc(ff_node *w) {
		return ((ffDenoiser<DenoiserKernelType> *) w)->getSvcTime();
	}

	void displayHeadersParallel(parallel_parameters_t *pp) {
		cout << "* ff parallel configuration:" << endl << "n. farm workers: "
				<< this->n_farm_workers << endl << "n. detector workers: "
				<< this->n_detector_kernel_workers << endl
				<< "n. denoiser workers: " << pp->n_denoiser_workers << endl
				<< "*" << endl;
	}

private:
	int n_denoiser_workers;
};

#endif /* FFDRIVER_HPP_ */
