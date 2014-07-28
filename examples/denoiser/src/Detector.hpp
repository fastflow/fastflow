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

#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_

#include <task_types.hpp>
#include <ff/node.hpp>
//#include <ff/parallel_for.hpp>

#include <ocv/BitmapInputOCV.hpp>
#include <ocv/BitmapOutputOCV.hpp>
#include <ocv/VideoInputOCV.hpp>
#include <ocv/CameraInputOCV.hpp>
#include <ocv/VideoOutputOCV.hpp>
#include <fastflow/spd/NoiserSPD.hpp>
#include <fastflow/gaussian/NoiserGaussian.hpp>

using namespace ff;

/*!
 * \class Detector
 *
 * \brief Detects noisy pixels
 *
 * This class detects noisy pixels according to some noise-specific detection kernel.
 */
template<typename DetectorKernelType>
class Detector: public ff_node {
public:

  Detector(unsigned int n_kernel_workers_, string in_fname, void *kernel_params, unsigned int n_frames_limit,
		  bool trace_time_, unsigned int noise_type, unsigned int noise, bool show_enabled) :
    n_kernel_workers(n_kernel_workers_), n_noisy(0), noisyMap(NULL), noisyPixels(NULL), trace_time(trace_time_),
    svc_time_us(0), n_frames_read(0), noisy_percent(0.0f) {
    //set file input
    if (isBitmap(in_fname))
      input = new BitmapInputOCV(in_fname);
    else
      if(strcmp(in_fname.data(), "VideoFromCamera.avi") == 0) {
	std::cerr << "Camera\n";
	input = new CameraInputOCV(in_fname);
      }
      else
	input = new VideoInputOCV(in_fname);



    //get frame info
    width = input->getWidth();
    height = input->getHeight();
    n_frames_total = input->getNFramesTotal();
    n_frames_work = std::min(n_frames_total, n_frames_limit);

    //set detector kernel
    kernel = new DetectorKernelType(kernel_params, height, width);

    //set (noisy) files output
    if (isBitmap(in_fname))
      outputNoisy = new BitmapOutputOCV("noisy", height, width);
    else
      outputNoisy = new VideoOutputOCV("noisy", input->getFPS(), height, width, show_enabled);

    //set noiser
    add_noise = noise > 0;
    if (noise_type == SPNOISE)
      noiser = new NoiserSPD(height, width, noise);
    else
      // Gauss
      noiser = new NoiserGaussian(height, width, 0, noise);
  }

  /*!
   * \brief Returns the noisy-map
   *
   * It returns the noisy-map built by detect() method.
   *
   * \return the noisy-map. The value of the pixel
   * (x,y) of the noisy-map is -1 if the corresponding
   * pixel is not noisy, otherwise it has the same value
   * as the original frame
   */
  int * getNoisyMap() {
    return noisyMap;
  }

  /*!
   * \brief Returns the array of noisy pixels
   *
   * It returns the array of noisy pixels built by detect() method.
   *
   * \return the array of (indexes of) noisy pixels
   */
  unsigned int * getNoisyPixels() {
    return noisyPixels;
  }

  /*!
   * \brief Returns the number of noisy pixels
   *
   * It returns the number of noisy pixels.
   *
   * \return the number of noisy pixels
   */
  unsigned int getNbNoisy() {
    return n_noisy;
  }

  unsigned long getSvcTime() {
    return svc_time_us;
  }

  float getNoisyPercent() {
    return noisy_percent;
  }

  /*!
   * \brief Detects noise over a frame
   *
   * It applies the AMF over a frame.
   *
   * \param noisyImage is the input frame
   */
  void detect(unsigned char * noisyImage) {
    unsigned long timestamp_us;
    if (trace_time)
      timestamp_us = get_usec_from(0);
    init(noisyImage);

    noisyMap = (int *) malloc(height * width * sizeof(int));

    //(MAP) parallel detect
    int chunk = std::max(height / n_kernel_workers, 1u);
    //FF_PARFOR_START(pf_det, ri, 0, height, 1, chunk, n_kernel_workers)
#pragma omp parallel for schedule(runtime) num_threads(n_kernel_workers)
    for (int ri=0;ri<height;++ri)
    {
      for (unsigned int ci = 0, x = ri * width; ci < width; ++ci, ++x)
	noisyMap[x] = ((*kernel)(noisyImage, noisyImage[x], ri, ci)) ? noisyImage[x] : -1;
    }
    //FF_PARFOR_STOP(pf_det);

    //(REDUCE) build array of noisy pixels
    n_noisy = 0;
    for (unsigned int ri = 0; ri < height; ++ri)
      for (unsigned int ci = 0, x = ri * width; ci < width; ++ci, ++x)
        if (noisyMap[x] >= 0)
          ++n_noisy;
    noisyPixels = (unsigned int *) malloc(n_noisy * sizeof(unsigned int));
    for (unsigned int i = 0, ri = 0; ri < height; ++ri)
      for (unsigned int ci = 0, x = ri * width; ci < width; ++ci, ++x)
        if (noisyMap[x] >= 0)
          noisyPixels[i++] = x;

    noisy_percent += ((float)n_noisy / (height * width));

    if (trace_time)
      svc_time_us += get_usec_from(timestamp_us);
  }

  int svc_init(){
    //FF_PARFOR_ASSIGN(pf_det,n_kernel_workers);
    return(0);
  }

  void svc_end(){
    //FF_PARFOR_DONE(pf_det);
  }

  void* svc(void * task_) {
    while (n_frames_read < n_frames_work) {
      //read
      unsigned char *in = input->nextFrame();
      ++n_frames_read;

      //add noise
      if (add_noise)
        noiser->addNoise(in);
      outputNoisy->writeFrame(in);

      //detect
      detect(in);

      //send out task to denoise stage
      denoise_task * task = new denoise_task();
      task->input = in;
      task->noisymap = getNoisyMap();
      task->noisy = getNoisyPixels();
      task->n_noisy = getNbNoisy();
      task->output = (unsigned char *) malloc(height * width * sizeof(unsigned char));
      task->height = height;
      task->width = width;
      ff_send_out(task);
      //cerr << "{"<<(void *)task->input<<"} detected " << (float) n_noisy / (height * width) * 100.0f << "% noisy" << endl;
    }
    return EOS ;
  }

  /*!
   * \brief Initializes the detection engine
   *
   * It initializes the detection engine for a frame.
   *
   * \param noisyImage is the input frame
   */
  virtual void init(unsigned char *noisyImage) {
  }

  virtual ~Detector() {

    if (input)
      delete input;
    if (kernel)
      delete kernel;
    if (noiser)
      delete noiser;
    if (outputNoisy)
      delete outputNoisy;
  }

  unsigned int getHeight() const {
    return height;
  }

  unsigned int getWidth() const {
    return width;
  }

  unsigned int getNFramesTotal() const {
    return n_frames_total;
  }

  unsigned int getNFramesWork() const {
    return n_frames_work;
  }

  ;

protected:
  unsigned int width;
  unsigned int height;
  int * noisyMap;
  unsigned int * noisyPixels;
  unsigned int n_noisy;
  bool trace_time;
  unsigned long svc_time_us;
  float noisy_percent;
private:
  unsigned int n_kernel_workers;
  DetectorKernelType *kernel;
  Output * outputNoisy;
  unsigned int n_frames_total, n_frames_work, n_frames_read;
  bool add_noise;
  Noiser * noiser;
  Input * input;
  //FF_PARFOR_DECL(pf_det);
};
#endif
