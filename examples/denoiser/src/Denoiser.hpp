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

#ifndef DENOISER_HPP_
#define DENOISER_HPP_

#define RESIDUAL_THRESHOLD 0.1f

#include <utils.hpp>

#include <task_types.hpp>
#include <ff/node.hpp>


using namespace ff;

/*!
 * \class Denoiser
 *
 * \brief Restores a frame
 *
 * This class restores noisy pixels of a frame according to some noise-specific restoration kernel.
 */
class Denoiser {
public:
  Denoiser(unsigned int height_, unsigned int width_, bool fixed_cycles_, unsigned int max_cycles_, bool trace_time_) :
      height(height_), width(width_), fixed_cycles(fixed_cycles_), max_cycles(max_cycles_), trace_time(trace_time_), svc_time_us(0), n_restored_frames(
																							    0), total_cycles(0) {
  }

  unsigned long getSvcTime() {
    return svc_time_us;
  }

  unsigned int getCycles() {
    return total_cycles;
  }

  unsigned int getNRestoredFrames() {
    return n_restored_frames;
  }

    virtual void svc_end() = 0;

  void* svc(void * task_) {
    denoise_task * task = (denoise_task *) task_;
    unsigned long timestamp_us;
    if (trace_time)
      timestamp_us = get_usec_from(0);
    task->n_cycles = restore(task->input, task->output, task->noisymap, task->noisy, task->n_noisy, task);
    if (trace_time)
      svc_time_us += get_usec_from(timestamp_us);
    total_cycles += task->n_cycles;
    ++n_restored_frames;
    //cerr << "{"<<(void *)task->input<<"} restored in " << task->n_cycles << " cycles" << endl;
    return task;
  }

  /*!
   * \brief Restores a frame
   *
   * It iterates the restoration kernel over a frame until
   * convergence is reached.
   *
   * \param in is the input frame
   * \param noisymap is the noisy-map
   * \param noisy is the array of noisy pixels
   * \param n_noisy is the length of the array of noisy pixels
   *
   * \return the number of restoration cycles
   */
  virtual unsigned int restore(unsigned char *in, unsigned char *out, int *noisymap, unsigned int *noisy, unsigned int n_noisy, void *task) = 0;

  virtual ~Denoiser() {}

protected:
  unsigned int height, width;
  bool fixed_cycles;
  unsigned int max_cycles;
  bool trace_time;
  unsigned long svc_time_us;
  unsigned int total_cycles;
  unsigned int n_restored_frames;
};
#endif
