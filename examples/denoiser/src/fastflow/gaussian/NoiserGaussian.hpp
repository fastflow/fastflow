/*
 * NoiserGauss.cpp
 *
 *  Created on: 13 janv. 2014
 *      Author: peretti
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

