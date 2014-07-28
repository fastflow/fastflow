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

#ifndef _DETECTORSPD_HPP_
#define _DETECTORSPD_HPP_

#include <defs.h>
#include <Detector.hpp>

struct DetectorKernelSPD_params {
  unsigned int w_max;
};

/*!
 * \class DetectorSPD
 *
 * \brief Check salt-and-pepper noise on a pixel
 *
 * This class detects salt-and-pepper noisy pixels
 * by applying a classical AMF.
 */
class DetectorKernelSPD {
public:

  DetectorKernelSPD(void *params_, unsigned int height_, unsigned int width_) :
  height(height_), width(width_) {
    DetectorKernelSPD_params *params = (DetectorKernelSPD_params *)params_;
    w_max = params->w_max;
  }

  bool operator()(unsigned char *im, unsigned char center, unsigned int r, unsigned int c) {
    return is_noisy(center, r, c, w_max, im, width, height);
  }

private:
  unsigned int height, width;
  unsigned int w_max;

  int ctrl_px(unsigned char *im, unsigned char *a, unsigned int w, unsigned int yy, unsigned int xx, unsigned int width,
      unsigned int height) {
    unsigned int cw(w / 2);
    int n = 0;
    for (unsigned int r = 0; r < w; r++) //rows
      for (unsigned int c = 0; c < w; c++) //columns
        if (((xx + c) >= cw) && ((xx + c) < (width + cw)) && ((yy + r) >= cw) && ((yy + r) < (height + cw)))
          a[n++] = im[(yy + r - cw) * width + xx + c - cw];
    return n;
  }

  //AMF filter on grayscale pixel
  inline bool is_noisy(unsigned char center, unsigned int r, unsigned int c, unsigned int w_max, unsigned char *im, unsigned int width,
      unsigned int height) {
    if (center == SALT || center == PEPPER) {
      unsigned char c_array[MAX_WINDOW_SIZE * MAX_WINDOW_SIZE];
      for (unsigned int ws = 3; ws < w_max; ws += 2) {
        int ac = ctrl_px(im, c_array, ws, r, c, width, height);
        sort(c_array, c_array + ac);
        unsigned char min_cp = c_array[0];
        unsigned char med_cp = c_array[ac / 2];
        unsigned char max_cp = c_array[ac - 1];
        if ((!(min_cp == med_cp && med_cp == max_cp)) || (ws > w_max)) //not homogeneous zone
          if (center == min_cp || center == max_cp || ws > w_max)
            return true; //noisy
      }
    }
    return false; //not noisy
  }
};
#endif
