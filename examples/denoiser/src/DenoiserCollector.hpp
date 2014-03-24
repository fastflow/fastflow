/*
 * DenoiserCollector.hpp
 *
 *  Created on: Jan 25, 2014
 *      Author: droccom
 */

#ifndef DENOISERCOLLECTOR_HPP_
#define DENOISERCOLLECTOR_HPP_

#include <utils.hpp>

#include <task_types.hpp>
#include <ff/node.hpp>
using namespace ff;

class DenoiserCollector : public ff_node {
public:
  DenoiserCollector(string in_fname, string out_fname, unsigned int height_, unsigned int width_, bool show_enabled) :
    height(height_), width(width_), nframes(0) {
      if (isBitmap(in_fname)) {
        output = new BitmapOutputOCV(out_fname, height, width);
      } else { //set video output
        cv::VideoCapture capture(in_fname);
        double fps;
        if(strcmp(in_fname.data(), "VideoFromCamera.avi") == 0)
        	fps=30.0;
        else
        	fps = capture.get(CV_CAP_PROP_FPS);

        output = new VideoOutputOCV(out_fname, fps, height, width, show_enabled);
      }
    }

  ~DenoiserCollector() {
    if(output)
      delete output;
  }

  void* svc(void * task_) {
      denoise_task * task = (denoise_task *) task_;
      output->writeFrame(task->output);
      cout << "\r" << nframes++ << flush;
      //clean-up
      delete (task);
      return GO_ON;
    }

private:
  unsigned int height, width;
  Output *output;
  unsigned int nframes;
};



#endif /* DENOISERCOLLECTOR_HPP_ */
