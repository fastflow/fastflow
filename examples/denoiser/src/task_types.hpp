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

#ifndef _TASK_TYPES_HPP_
#define _TASK_TYPES_HPP_

//denoise task
typedef struct denoise_task {
  ~denoise_task() { //TODO: check delete, delete[], free
    if(input) delete input;
    if(output) delete output;
    if(noisymap) delete noisymap;
    if(noisy) delete noisy;
  }

  denoise_task() {
	  input = output = NULL;
	  noisymap = NULL;
	  noisy = NULL;
	  n_noisy = n_cycles = width = height = 0;
  }

  unsigned char *input, *output;
  int *noisymap; //-1 if not noisy, otherwise the original pixel
  unsigned int *noisy;
  unsigned int n_noisy;
  unsigned int n_cycles;
  unsigned int width, height;
  void *kernel_params;
  bool fixed_cycles;
} denoise_task_t;

#endif
