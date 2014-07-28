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
