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

#ifndef _VIDEOINPUTOCV_HPP_
#define _VIDEOINPUTOCV_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <ocv/cast.hpp>
#include <Input.hpp>

/*!
 * \class VideoInputOCV
 *
 * \brief Reads frames from a video stream using OpenCV
 *
 * This class reads frames from a video stream using OpenCV.
 */
class VideoInputOCV: public Input {
public:
  VideoInputOCV(string fname) {
    //get frame info
	capture = cv::VideoCapture(fname);
	nFramesTotal = (unsigned int) capture.get(CV_CAP_PROP_FRAME_COUNT);
	fps = capture.get(CV_CAP_PROP_FPS);
    width = (unsigned int) capture.get(CV_CAP_PROP_FRAME_WIDTH);
    height = (unsigned int) capture.get(CV_CAP_PROP_FRAME_HEIGHT);
  }

  unsigned char *nextFrame() {
    //read frame
    cv::Mat frame;
    capture >> frame;
    //convert to grayscale
    cv::Mat frame_grayscale(cv::Size((int) width, (int) height), CV_8UC1);
    cv::cvtColor(frame, frame_grayscale, CV_RGB2GRAY);
    return Mat2uchar<unsigned char>(frame_grayscale);
  }

private:
  cv::VideoCapture capture;
};
#endif
