/*
 * BitmapInput.hpp
 *
 *  Created on: 19 janv. 2014
 *      Author: peretti
 */

#ifndef BITMAPINPUTOCV_HPP_
#define BITMAPINPUTOCV_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <Input.hpp>
#include <ocv/cast.hpp>

using namespace cv;

/*!
 * \class BitmapInputOCV
 *
 * \brief Reads a bitmap file using OpenCV
 *
 */
class BitmapInputOCV: public Input {
public:
  BitmapInputOCV(string fname) {
    //get frame info
    frame = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file (grayscale)
    width = (unsigned int) frame.cols;
    height = (unsigned int) frame.rows;
    nFramesTotal = 1;
  }

  unsigned char *nextFrame() {
    return Mat2uchar<unsigned char>(frame);
  }

private:
  Mat frame;
};

#endif /* BITMAPINPUTOCV_HPP_ */
