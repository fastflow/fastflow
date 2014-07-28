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

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

//#include <utils.hpp>

#include <string>
#include <limits>
using namespace std;

#define OUTFILE_EXT "avi"

#define MAX_CYCLES 2000
#define MAX_CYCLES_DEFAULT 200
#define NFRAMES_DEFAULT std::numeric_limits<int>::max()
#define MAX_WINDOW_SIZE (39*2+1)
#define ALFA_DEFAULT 1.3f
#define BETA_DEFAULT 5.0f

#define FARM_WORKERS_DEFAULT 1
#define DETECTOR_WORKERS_DEFAULT 1
#define DENOISER_WORKERS_DEFAULT 1

#define SPNOISE 1
#define GAUSSNOISE 2

/*!
 * \struct arguments
 *
 * \brief command line arguments
 */
typedef struct arguments {
  float alfa, beta;
  int w_max, max_cycles, noise, nframes, noise_type;
  bool fixed_cycles;
  string fname, out_fname, conf_fname;
  bool verbose, show_enabled, user_out_fname, add_noise;
} arguments;

struct parallel_parameters_t {
  parallel_parameters_t() {
    n_farm_workers = FARM_WORKERS_DEFAULT;
    n_detector_workers = DETECTOR_WORKERS_DEFAULT;
    n_denoiser_workers = DENOISER_WORKERS_DEFAULT;
  }
  unsigned int n_farm_workers, n_detector_workers, n_denoiser_workers;
};

void print_help();
void get_arguments(char *argv[], int argc, arguments &args);
void get_parallel_parameters(arguments &args, parallel_parameters_t *parameters);
#endif
