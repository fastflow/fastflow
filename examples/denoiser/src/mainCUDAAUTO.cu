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

#define FF_CUDA

#include <parameters.h>
#include <utils.hpp>
#include <Input.hpp>
#include <Output.hpp>
#include <Detector.hpp>
#include <DenoiserCollector.hpp>
#include <cuda/cudaDriverAUTO.hpp>
#include <cuda/spd/DenoiserCUDATaskSPD.hpp>
#include <cuda/spd/DenoiserMapFSPD.hpp>
#include <cuda/gaussian/DenoiserCUDATaskGaussian.hpp>
#include <cuda/gaussian/DenoiserMapFGaussian.hpp>
#include <fastflow/spd/DetectorKernelSPD.hpp>
#include <fastflow/gaussian/DetectorKernelGaussian.hpp>
#include <task_types.hpp>

#include <string>

#include <ff/pipeline.hpp>
#include <ff/farm.hpp>

using namespace ff;

/*!
 * \mainpage
 * Restores images/videos affected by Salt&Pepper / Gaussian noise
 *
 */

int main(int argc, char *argv[]) {
  arguments args;
  get_arguments(argv, argc, args);

  parallel_parameters_t pp;
  get_parallel_parameters(args, &pp);

  bool trace_time = true;

  if (args.noise_type == SPNOISE) {
    cudaDriverAUTO<DetectorKernelSPD, DenoiserCUDATaskSPD, DenoiserMapFSPD> driver(args, trace_time);
    DetectorKernelSPD_params detector_kernel_params;
    detector_kernel_params.w_max = args.w_max;
    DenoiserKernelSPD_params denoiser_kernel_params;
    denoiser_kernel_params.alfa = args.alfa;
    denoiser_kernel_params.beta = args.beta;
    driver.init(&args, &pp, &detector_kernel_params, &denoiser_kernel_params);
    driver.goFF();
    if (trace_time)
      driver.printTimes(std::cout);
  }
    else { //Gaussian
    cudaDriverAUTO<DetectorKernelGaussian, DenoiserCUDATaskGaussian, DenoiserMapFGaussian> driver(args, trace_time);
    DenoiserKernelGaussian_params denoiser_kernel_params;
    denoiser_kernel_params.alfa = args.alfa;
    denoiser_kernel_params.beta = args.beta;
    driver.init(&args, &pp, NULL, &denoiser_kernel_params);
    driver.goFF();
    if (trace_time)
      driver.printTimes(std::cout);
  }

  cout << "done" << endl;
  return 0;
}
