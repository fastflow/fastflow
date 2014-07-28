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

#ifndef DRIVER_HPP_
#define DRIVER_HPP_

#include <string>
#include <vector>

#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>

#include <parameters.h>
#include <Denoiser.hpp>

using namespace std;
using namespace ff;

/*!
 * \class Driver
 *
 * \brief Orchestrates all the image denoising steps (input/output, detection, denoising)
 *
 * This class orchestrates all the image denoising steps (input/output, detection, denoising)
 *
 */
template<typename DetectorKernelType>
class Driver {
public:
  Driver(arguments args, bool trace_time_ = false) :
      in_fname(args.fname), alfa(args.alfa), beta(args.beta), w_max(args.w_max), max_cycles(args.max_cycles), fixed_cycles(
          args.fixed_cycles), user_out_fname(args.user_out_fname), verbose(args.verbose), show_enabled(args.show_enabled), noise(
          args.noise), noise_type(args.noise_type), n_frames(args.nframes), n_frames_total(0), n_frames_work(0), height(0), width(0), detector(
      NULL), trace_time(trace_time_), exec_time_us(0), denoiser_collector(NULL),
      n_farm_workers(0), n_detector_kernel_workers(0) {
    add_noise = noise > 0;

    if (user_out_fname)
      out_fname = args.out_fname;

    if (user_out_fname)
      prefix = out_fname.substr(0, out_fname.find_last_of("."));
    else {
      prefix.append("RESTORED_");
      string trunked_fname = get_fname(in_fname);
      prefix.append(trunked_fname.substr(0, trunked_fname.find_last_of(".")));
    }
    out_fname = prefix;
  }

  virtual void displayHeadersParallel(parallel_parameters_t *) = 0;

  void init(arguments *args, parallel_parameters_t *pp,
      void *detector_kernel_params, void *denoiser_kernel_params) {
    n_farm_workers = pp->n_farm_workers;
    n_detector_kernel_workers = pp->n_detector_workers;
    //n_denoiser_kernel_workers = pp->n_denoiser_workers;

    //set detector
    detector = new Detector<DetectorKernelType>(n_detector_kernel_workers, in_fname, detector_kernel_params, n_frames, trace_time,
            noise_type, noise, show_enabled);
    //detector->setNodeType(ARBITER); //experimental

    //get frame info
    width = detector->getWidth();
    height = detector->getHeight();
    n_frames_total = detector->getNFramesTotal();
    n_frames_work = detector->getNFramesWork();

    //set denoiser farm
    for (unsigned int i = 0; i < n_farm_workers; ++i)
      denoiser_workers.push_back(createDenoiserNode(pp, denoiser_kernel_params));
    denoiser_farm.add_workers(denoiser_workers);
    denoiser_collector = new DenoiserCollector(in_fname, out_fname, height, width, show_enabled);
    denoiser_farm.setCollectorF(denoiser_collector);

    //set pipe
    pipe.add_stage(detector);
    pipe.add_stage(&denoiser_farm);
    //pipe.optimise_pinning(); //experimental

    if (verbose)
      displayHeaders();
      displayHeadersParallel(pp);
  }

  virtual ~Driver() {
    if (detector)
      delete detector;
    if (denoiser_collector)
      delete denoiser_collector;
  }

  void goFF() {
    unsigned long timestamp_usec;
    if (trace_time)
      timestamp_usec = get_usec_from(0);

    pipe.run_and_wait_end();

    if (trace_time)
      exec_time_us = get_usec_from(timestamp_usec);
  }

virtual unsigned int get_worker_cycles(ff_node *) = 0;
virtual double get_worker_svc(ff_node *) = 0;

  void printTimes(std::ostream &os) {
    os << "mean noise: " << 100.0f * detector->getNoisyPercent() / n_frames_work << " %\n";
    float avg_cycles = 0.0f;
    for (unsigned int i = 0; i < n_farm_workers; ++i)
      avg_cycles += get_worker_cycles(denoiser_workers[i]);
    os << "mean cycles: " << avg_cycles / n_frames_work << endl;
    os << "overall execution time: " << (double) exec_time_us / 1e06 << " s" << std::endl << "overall throughput: "
        << double(1e06f) * n_frames_work / exec_time_us << " fps" << std::endl << "DETECT mean svc time: "
        << (double) detector->getSvcTime() / (n_frames_work * 1e03) << " ms" << std::endl << "DENOISE mean svc time: ";
    double denoiser_svc_time = 0;
    for (unsigned int i = 0; i < n_farm_workers; ++i)
      denoiser_svc_time += get_worker_svc(denoiser_workers[i]);
    os << denoiser_svc_time / (n_frames_work * 1e03) << " ms" << std::endl;
    printTimesRow(std::cerr);
  }

  void printTimesRow(std::ostream &os) {
    os << height << ", "
       << width << ", "
       << n_frames_work << ", "
       << 100.0f * detector->getNoisyPercent() / n_frames_work << ", "
       << n_farm_workers << ", "
       << n_detector_kernel_workers << ", "
       //<< n_denoiser_kernel_workers << ", "
       << (double) exec_time_us / 1e06 << ", "
       << double(1e06f) * n_frames_work / exec_time_us << ", "
       << (double) detector->getSvcTime() / (n_frames_work * 1e03) << ", ";
    double denoiser_svc_time = 0;
    for (unsigned int i = 0; i < n_farm_workers; ++i)
      denoiser_svc_time += get_worker_svc(denoiser_workers[i]);
    os << denoiser_svc_time / (n_frames_work * 1e03);
    os << endl;
  }

protected:
  virtual ff_node *createDenoiserNode(parallel_parameters_t *, void *) = 0;
  int n_farm_workers, n_detector_kernel_workers;
  unsigned int width, height;
  unsigned int max_cycles;
  bool fixed_cycles, trace_time;

private:
  void displayHeaders() {
    cout << "*** This is Denoiser Deluxe" << endl << "noise type = " << ((noise_type == SPNOISE) ? "Salt & Pepper " : "Gaussian ") << endl
        << "control-window size: " << w_max << endl << "alpha = " << alfa << "; beta = " << beta << endl << "max number of cycles = "
        << max_cycles << endl;
    if (fixed_cycles)
      cout << "number of cycles fixed to " << max_cycles << endl;
    if (add_noise) {
      if (noise_type == SPNOISE)
        cout << "will add " << noise << "% of S&P noise" << endl;
      else
        cout << "will add gaussian noise with variance " << noise << " and mean 0" << endl;
    }
    cout << "Input: " << in_fname << endl << "Output: " << prefix << endl << "---" << endl;
    cout << n_frames_total << " frames ( " << width << " x " << height << " )" << endl;
    if (n_frames_work < n_frames_total)
      cout << n_frames_work << " frames will be restored" << endl;
  }

  string in_fname, out_fname;
  float alfa, beta;
  unsigned int w_max, noise, n_frames, noise_type;
  bool verbose, show_enabled, add_noise, user_out_fname;
  string prefix;
  unsigned int n_frames_total, n_frames_work;
  //building blocks
  ff_pipeline pipe;
  Detector<DetectorKernelType> * detector;
  ff_ofarm denoiser_farm;
  DenoiserCollector *denoiser_collector;
  vector<ff_node *> denoiser_workers;
  //time measurement
  unsigned long exec_time_us;
};

#endif /* DRIVER_HPP_ */
