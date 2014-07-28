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

#ifndef _DENOISERSPD_HPP_
#define _DENOISERSPD_HPP_

#include <Denoiser.hpp>
#include <fastflow/spd/kernel_cpp/fuy.hpp>
#include <fastflow/spd/kernel_cpp/pow_table.hpp>

struct DenoiserKernelSPD_params {
  float alfa, beta;
};

/*!
 * \class DenoiserSPD
 *
 * \brief Restored a frame affected by salt-and-pepper noise
 *
 * This class restores a frame affected by salt-and-pepper noisy
 * by applying the modified Chan variational method.
 */
class DenoiserKernelSPD {
public:
  DenoiserKernelSPD(void *params_, unsigned int height_, unsigned int width_) :
    height(height_), width(width_) {
DenoiserKernelSPD_params *params = (DenoiserKernelSPD_params *)params_;
    alfa = params->alfa;
    beta = params->beta;
    pt = new pow_table(alfa);
  }

  ~DenoiserKernelSPD() {
    delete pt;
  }

  /*!
   * \brief Restores a pixel
   *
   * It applies the modified Chan kernel over a pixel.
   *
   * \param in is the input frame
   * \param x is the index of the pixel to be restored
   * \param noisymap is the noisy-map
   *
   * \return the value of the restored pixel
   */
  unsigned char operator()(unsigned char *in, unsigned int x, int *noisymap) {
    return fuy(in, x, noisymap, width, height, alfa, beta, *pt);
  }

  void restore_chunk(unsigned int *noisy, unsigned int first, unsigned int last, int *noisymap, unsigned char *in, unsigned char *out) {
    for (int i=first; i<=last; ++i)
      {
	unsigned int x = noisy[i];
        out[x] = fuy(in, x, noisymap, width, height, alfa, beta, *pt);
      }
  }

private:
  unsigned int height, width;
  float alfa, beta;
  pow_table *pt;
};
#endif
