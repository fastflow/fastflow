#ifndef _NOISERSPD_HPP_
#define _NOISERSPD_HPP_

#include <Noiser.hpp>
#include <defs.h>

/*!
 * \class NoiserSPD
 *
 * \brief Adds salt-and-pepper noise to a frame
 *
 * This class adds salt-and-pepper noise to a frame.
 */
class NoiserSPD: public Noiser {
public:
  NoiserSPD(unsigned int height, unsigned int width, double noise_) :
      Noiser(height, width), noise(noise_) {
  }

  /*!
   * \brief Adds salt-and-pepper noise to a frame
   *
   * It adds salt-and-pepper noise to a frame.
   *
   * \param im is the input image
   */
  void addNoise(unsigned char *im) {
    addSaltandPepperNoiseDEPTH_8U_cv2(im, noise, height, width);
  }

private:
  double noise;

  void addSaltandPepperNoiseDEPTH_8U_cv2(unsigned char *img, double percent, unsigned int height, unsigned int width) {
    double pr = 1.0 - percent / 100.0;
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j) {
        // Generate random number between -1.0 and +1.0
        double rr = (double) rand();
        double random = 2.0 * (rr - (RAND_MAX) / 2.0) / (RAND_MAX);
        if (random > pr)
          img[i * width + j] = (unsigned char) SALT;
        else if (random < -pr)
          img[i * width + j] = (unsigned char) PEPPER;
      }
  }
};
#endif
