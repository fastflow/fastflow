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

#ifndef __FUY_HPP__
#define __FUY_HPP__

#include "pow_table.hpp"
#include <cmath>
#include <cfloat>

inline unsigned char fuy(const unsigned char *im, const unsigned int idx, const int *noisymap, const unsigned int w, const unsigned int h,
    const float alfa, const float beta, pow_table &pt) {
  //get the pixel and the four closest neighbors (with replication-padding)
  unsigned char pixel = im[idx];
  //up
  int idx_neighbor = idx - w * (idx >= w);
  unsigned char up_val = im[idx_neighbor];
  unsigned char up_noisy = (noisymap[idx_neighbor] >= 0);
  //down
  idx_neighbor = idx + w * (idx < ((h - 1) * w));
  unsigned char down_val = im[idx_neighbor];
  unsigned char down_noisy = (noisymap[idx_neighbor] >= 0);
  //left
  idx_neighbor = idx - ((idx % w) > 0);
  unsigned char left_val = im[idx_neighbor];
  unsigned char left_noisy = (noisymap[idx_neighbor] >= 0);
  //right
  idx_neighbor = idx + ((idx % w) < (w - 1));
  unsigned char right_val = im[idx_neighbor];
  unsigned char right_noisy = (noisymap[idx_neighbor] >= 0);

  //compute the correction
  unsigned char u;
  float S;
  float Fu, u_min = 0.0f, Fu_prec = FLT_MAX;
  float beta_ = beta / 2;
  for (int uu = 0; uu < 256; ++uu) {
    u = (unsigned char) uu;
    Fu = 0.0f;
    S = 0.0f;
    S += (float) (2 - up_noisy) * pt(_ABS(uu - (int) up_val));
    S += (float) (2 - down_noisy) * pt(_ABS(uu - (int) down_val));
    S += (float) (2 - left_noisy) * pt(_ABS(uu - (int) left_val));
    S += (float) (2 - right_noisy) * pt(_ABS(uu - (int) right_val));
    Fu += _ABS((float) u - (float) pixel) + (beta_) * S;
    if (Fu < Fu_prec) {
      u_min = u;
      Fu_prec = Fu;
    }
  }

  unsigned char new_val = (unsigned char) (u_min + 0.5f); //round
  return new_val;
}
#endif 
