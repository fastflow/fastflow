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

#ifndef _CAMERAINPUTOCV_HPP_
#define _CAMERAINPUTOCV_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <ocv/cast.hpp>
#include <Input.hpp>
#include <opencv/cvaux.h>

using namespace cv;
/*!
 * \class CameraInputOCV
 *
 * \brief Reads frames from a video stream using OpenCV
 *
 * This class reads frames from a video stream using OpenCV.
 */
class CameraInputOCV: public Input {
public:
	CameraInputOCV(string fname) {
		//get frame info
		//		capture = cv::VideoCapture(0);
		//	 	width = (unsigned int) capture.get(CV_CAP_PROP_FRAME_WIDTH);
		//	    height = (unsigned int) capture.get(CV_CAP_PROP_FRAME_HEIGHT);
		//		if (!capture.isOpened()) {
		//			cout << "Input file not given and cannot open the video cam" << endl;
		//			exit(1);
		//		}
	  std::cerr << "Constructor\n";
	  capture = cvCreateCameraCapture(CV_CAP_ANY);
	  if (!capture) {
	    printf("Error: Cannot open the webcam !\n");
	    exit(0);
	  }
//	  cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH, 426);
//	  cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT, 320);
	  width = (int) cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH);
	  height = (int) cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT);
	  std::cerr << " WxH " << width << "x" << height << "\n";
	  fps=30.0;
	}

  unsigned char *nextFrame() {
	  //read frame
	  IplImage * image = cvQueryFrame(capture);
	  Mat frame(image);
//    capture >> frame;

    //convert to grayscale
    Mat frame_grayscale(Size((int) width, (int) height), CV_8UC1);
    cvtColor(frame, frame_grayscale, CV_RGB2GRAY);
    return Mat2uchar<unsigned char>(frame_grayscale);
  }

private:
//  cv::VideoCapture capture;
  CvCapture * capture;
};
#endif
