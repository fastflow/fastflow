/*
 * cudaDriver.hpp
 *
 *  Created on: Feb 24, 2014
 *      Author: droccom
 */

#ifndef CUDADRIVERAUTO_HPP_
#define CUDADRIVERAUTO_HPP_

#include <Driver.hpp>
#include <cuda/cudaDenoiserAUTO.hpp>

template<typename DetectorKernelType, typename DenoiserCUDAtaskType, typename DenoiserCUDAmapF>
class cudaDriverAUTO : public Driver<DetectorKernelType> {
public:
	cudaDriverAUTO(arguments args, bool trace_time = false) : Driver<DetectorKernelType>(args, trace_time) {}

  ff_node *createDenoiserNode(parallel_parameters_t *pp, void *denoiser_kernel_params) {
    cudaDenoiserAUTO<DenoiserCUDAtaskType, DenoiserCUDAmapF> *res = new cudaDenoiserAUTO<DenoiserCUDAtaskType, DenoiserCUDAmapF>(denoiser_kernel_params, this->height, this->width, this->fixed_cycles, this->max_cycles, this->trace_time);
    res->setMaxThreads(512);
    return res;
  }

  unsigned int get_worker_cycles(ff_node *w) {
      return ((cudaDenoiserAUTO<DenoiserCUDAtaskType, DenoiserCUDAmapF> *)w)->getCycles();
    }

    double get_worker_svc(ff_node *w) {
      return ((cudaDenoiserAUTO<DenoiserCUDAtaskType, DenoiserCUDAmapF> *) w)->getSvcTime();
    }


    void displayHeadersParallel(parallel_parameters_t *pp) {
        cout << "* cuda parallel configuration:" << endl
            << "n. farm workers: " << this->n_farm_workers << endl
            << "n. detector workers: " << this->n_detector_kernel_workers << endl
            << "*" << endl;
      }
};

#endif /* CUDADRIVER_HPP_ */
