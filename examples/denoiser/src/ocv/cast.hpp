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

#ifndef _SPD_CAST_HPP_
#define _SPD_CAST_HPP_

#include <opencv2/opencv.hpp>

//#include <iostream>
//using namespace std;

template<typename T>
T *Mat2uchar(cv::Mat &in) {
  T *out = new T[in.rows * in.cols];
  //std:cerr << " RxC " << in.rows << "x" << in.cols;
  for (int i = 0; i < in.rows; ++i)
    for (int j = 0; j < in.cols; ++j)
      out[i * (in.cols) + j] = in.at<T>(i, j);
  return out;
}

//useless: cv::Mat constructor does it!
//template<typename T>
//cv::Mat uchar2Mat(T *in, unsigned int height, unsigned int width) {
//  cv::Mat im(cvSize(width, height), CV_8UC1);
//  for (unsigned int i = 0; i < height; ++i)
//    for (unsigned int j = 0; j < width; ++j)
//      im.at<uchar>(i, j) = (unsigned char) (in[i * width + j]);
//  return im;
//}
#endif
