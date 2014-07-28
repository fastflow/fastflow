#include <math.h>
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

#include <float.h>
//#include <cmath>
//#include <cfloat>

__global__ void fuy(unsigned char *in, unsigned char *out, int *noisymap, unsigned int *noisy, unsigned int n_noisy
		, unsigned int w, unsigned int h, float alfa, float beta) {

	int idx;
	if(blockIdx.x*blockDim.x + threadIdx.x < n_noisy)
		idx = noisy[blockIdx.x*blockDim.x + threadIdx.x];

	if(idx<h*w){
		out[idx]=idx;
		//get the pixel and the four closest neighbors (with replication-padding)
		unsigned char pixel = in[idx];
		//up
		int idx_neighbor = idx - w * (idx >= w);
		unsigned char up_val = in[idx_neighbor];
		unsigned char up_noisy = (noisymap[idx_neighbor] >= 0);
		//down
		idx_neighbor = idx + w * (idx < ((h - 1) * w));
		unsigned char down_val = in[idx_neighbor];
		unsigned char down_noisy = (noisymap[idx_neighbor] >= 0);
		//left
		idx_neighbor = idx - ((idx % w) > 0);
		unsigned char left_val = in[idx_neighbor];
		unsigned char left_noisy = (noisymap[idx_neighbor] >= 0);
		//right
		idx_neighbor = idx + ((idx % w) < (w - 1));
		unsigned char right_val = in[idx_neighbor];
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
			S += (float) (2 - up_noisy) * __powf(abs(uu - (int) up_val), alfa);
			S += (float) (2 - down_noisy) * __powf(abs(uu - (int) down_val), alfa);
			S += (float) (2 - left_noisy) * __powf(abs(uu - (int) left_val), alfa);
			S += (float) (2 - right_noisy) * __powf(abs(uu - (int) right_val), alfa);
			Fu += abs((float) u - (float) pixel) + (beta_) * S;
			if (Fu < Fu_prec) {
				u_min = u;
				Fu_prec = Fu;
			}
		}
		out[idx]= (unsigned char) (u_min + 0.5f); //round
	}
}

void kernel_wrapper(unsigned char *d_in, unsigned char *d_out, int *d_noisyMap, unsigned int *noisy, unsigned int n_noisy,
		unsigned int w, unsigned int h, float alfa, float beta, int nBlocks, int threadsPerBlock ){
	fuy<<<nBlocks, threadsPerBlock>>>(d_in, d_out, d_noisyMap, noisy, n_noisy, w, h, alfa, beta );

}
