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

#ifndef DENOISERMAPFGAUSSIAN_HPP_
#define DENOISERMAPFGAUSSIAN_HPP_

#include <ff/stencilReduceCUDA.hpp>

//#include <ff/map.hpp>
using namespace ff;

FFMAPFUNC6(DenoiserMapFGaussian, float, unsigned int, idx, unsigned char, im, unsigned char, out, int, noisymap, float, params, unsigned int, bitmap_size, float, diff,
		int tid = blockIdx.x*blockDim.x + threadIdx.x;

		float alfa = params[0];
		float beta = params[1];
		unsigned int h = bitmap_size[0];
		unsigned int w = bitmap_size[1];

		//get the pixel and the 8 closest
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
		//up-left
		idx_neighbor = idx - 1 - w * (idx >= w);
		unsigned char upl_val = im[idx_neighbor];
		unsigned char upl_noisy = (noisymap[idx_neighbor] >= 0);
		//up-right
		idx_neighbor = idx + 1 - w * (idx >= w);
		unsigned char upr_val = im[idx_neighbor];
		unsigned char upr_noisy = (noisymap[idx_neighbor] >= 0);
		//down-left
		idx_neighbor = idx - 1 + w * (idx < ((h - 1) * w));
		unsigned char downl_val = im[idx_neighbor];
		unsigned char downl_noisy = (noisymap[idx_neighbor] >= 0);
		//down-right
		idx_neighbor = idx + 1 + w * (idx < ((h - 1) * w));
		unsigned char downr_val = im[idx_neighbor];
		unsigned char downr_noisy = (noisymap[idx_neighbor] >= 0);

		//compute the correction
		unsigned char u = 0;
		float S;
		float Fu;
		float u_min = 0.0f;
		float Fu_prec = FLT_MAX; // 256.0f;
		float beta_ = beta; // / 2;
		for (int uu = 0; uu < 256; ++uu) {
			u = (unsigned char) uu;
			Fu = 0.0f;
			S = 0.0f;
			S += (float) (2 - up_noisy) * sqrt(abs((int) u - (int) up_val) * abs((int) u - (int) up_val) + alfa);
			S += (float) (2 - down_noisy) * sqrt(abs(((int) u - (int) down_val)) * abs(((int) u - (int) down_val)) + alfa);
			S += (float) (2 - left_noisy) * sqrt(abs(((int) u - (int) left_val)) * abs(((int) u - (int) left_val)) + alfa);
			S += (float) (2 - right_noisy) * sqrt(abs(((int) u - (int) right_val)) * abs(((int) u - (int) right_val)) + alfa);
			S += (float) (2 - upl_noisy) * sqrt(abs((int) u - (int) upl_val) * abs((int) u - (int) upl_val) + alfa);
			S += (float) (2 - upr_noisy) * sqrt(abs((int) u - (int) upr_val) * abs((int) u - (int) upr_val) + alfa);
			S += (float) (2 - downl_noisy) * sqrt(abs(((int) u - (int) downl_val)) * abs(((int) u - (int) downl_val)) + alfa);
			S += (float) (2 - downr_noisy) * sqrt(abs(((int) u - (int) downr_val)) * abs(((int) u - (int) downr_val)) + alfa);

			Fu = ((float) abs(u - pixel) + (beta_ * S));
			if (Fu < Fu_prec) {
				u_min = u;
				Fu_prec = Fu;
			}
		}
		out[idx] = (unsigned char) (u_min + 0.5f); //round
		unsigned char newdiff = (unsigned char) (abs(
				(int) (out[idx]) - noisymap[idx]));
		unsigned char res = (float) (abs((int) newdiff - (int) (diff[tid])));
		diff[tid] = newdiff;

		return res;
		);

#endif /* DENOISERMAPFGAUSSIAN_HPP_ */
