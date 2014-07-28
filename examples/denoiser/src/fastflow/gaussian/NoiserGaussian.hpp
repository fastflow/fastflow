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

#include "Noiser.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#ifndef NOISERGAUSS_H_
#define NOISERGAUSS_H_

using namespace cv;

/*!
 * \class NoiserGaussian
 *
 * \brief Adds gaussian noise to a frame
 *
 */
class NoiserGaussian: public Noiser {
public:
  NoiserGaussian(unsigned int height, unsigned int width, int m, float v) :
      Noiser(height, width), mean(m), variance(v) {
  }

  void addNoise(unsigned char *im) {
    addGaussianNoise(im);
  }

private:
  int mean;
  float variance;

  void addGaussianNoise(unsigned char *im) {
    Mat orig(height, width, CV_8U, im, Mat::AUTO_STEP); // convert uc* to Mat
    Mat noise(height, width, CV_8U);
    Mat noisy(height, width, CV_8U);

    RNG rng(0x12345);
    rng.fill(noise, RNG::NORMAL, mean, variance);
    add(orig, noise, noisy, noArray(), CV_8U);
    noisy.copyTo(orig);
  }
};

#endif /* NOISERGAUSS_H_ */

