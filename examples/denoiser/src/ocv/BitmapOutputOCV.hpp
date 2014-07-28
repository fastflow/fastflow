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

#ifndef _OCV_BITMAP_OUTPUT_HPP_
#define _OCV_BITMAP_OUTPUT_HPP_

#include <Output.hpp>

#include <opencv2/opencv.hpp>
#include <string>

/*!
 * \class BitmapOutputOCV
 *
 * \brief Writes bitmap to a file using OpenCV
 *
 */
class BitmapOutputOCV: public Output {
public:
  BitmapOutputOCV(string out_fname, int height, int width) :
      Output(height, width), fname(out_fname) {
    fname.append(".bmp");
  }
  ;

  void writeFrame(unsigned char *frame) {
    imwrite(fname, cv::Mat(height, width, CV_8U, frame, cv::Mat::AUTO_STEP));
  }

private:
  string fname;
};
#endif
