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
