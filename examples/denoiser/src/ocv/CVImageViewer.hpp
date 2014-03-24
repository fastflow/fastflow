/*
 * CVImageViewer.h
 *
 *  Created on: 14 janv. 2014
 *      Author: peretti
 */

#ifndef CVIMAGEVIEWER_HPP_
#define CVIMAGEVIEWER_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>

using namespace cv;

/*!
 * \class CVImageViewer
 *
 * \brief Displays a frame on the screen
 *
 * This class a frame on the screen
 *
 */
class CVImageViewer {
public:
  CVImageViewer(unsigned int height_, unsigned int width_) :
      height(height_), width(width_) {
  }

  /**
   *
   * @param im image to display
   * @param windowTitle window title
   */
  void show(unsigned char *im, string windowTitle) {
    Mat test(height, width, CV_8U, im, Mat::AUTO_STEP);
    namedWindow(windowTitle, WINDOW_AUTOSIZE); // Create a window for display.
    imshow(windowTitle, test);                   // Show our image inside it.
  }

private:
  unsigned int height;
  unsigned int width;
};

#endif /* CVIMAGEVIEWER_H_ */
