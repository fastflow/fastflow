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
