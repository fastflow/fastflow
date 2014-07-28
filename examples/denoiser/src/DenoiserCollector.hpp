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
