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

#include "parameters.h"
#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && defined(_WIN32)
#include "XGetopt/XGetopt.h"
#else
#include <getopt.h>
#endif

#include <string>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <fstream>
#include <unistd.h>
using namespace std;

//funzioni per il cheking dei parametri
bool err_w(int w) {
	return ((w < 3) || (w > MAX_WINDOW_SIZE)|| (w % 2 == 0));
}

bool err_alfa(float a) {
	return !(a > 0);
}

bool err_beta(float b) {
	return !(b > 0);
}

bool err_max_cycles(int c) {
	return !(c <= MAX_CYCLES && c >= 1);
}

bool err_nframes(int x) {
	return !(x > 0);
}

bool err_noise_type(int noiseType) {
	return !(noiseType == SPNOISE || noiseType == GAUSSNOISE);
}

// bool err_bmp(string &nomefile) {
//   unsigned long  l = nomefile.length();
//   return l<5 || (string("bmp").compare(nomefile.substr(l - 3, 3)) != 0);
// }

bool err_noise(int noise) {
	return !(noise >= 1 /*&& noise <= 99*/);
}

bool err_workers(int w) {
	return !(w > 0);
}

void print_help(char *exe) {
	cout << "Usage: " << exe << " [options] <file>" << endl;
	cout << "Input is taken from camera if no <file> is specified" << endl;
	cout << "Allowed options:" << endl << "-h\t\tthis help message" << endl
			<< "-N arg\t\tnoise type (" << SPNOISE << " = Salt & Pepper / "
			<< GAUSSNOISE << " = Gaussian) " << endl << "-v\t\tverbose mode"
			<< endl << "-s\t\tshow input and output bitmaps" << endl
			<< "-n arg\t\tnoise-% (1 <= arg <= 99)" << endl
			<< "-a arg\t\talpha (1 < arg <= 2) " << "[default: " << ALFA_DEFAULT
			<< "]" << endl << "-b arg\t\tbeta (0.5 <= arg <= 10) "
			<< "[default: " << BETA_DEFAULT << "]" << endl
			<< "-w arg\t\tcontrol-window size (3 <= arg <= " << MAX_WINDOW_SIZE
			<< ") " << "[default: " << MAX_WINDOW_SIZE << "]" << endl
			<< "-c arg\t\tmax cycles of restoration (1 <= arg <= 2000) "
			<< "[default: " << MAX_CYCLES_DEFAULT << "]" << endl
			<< "-f\t\tfix n. of cycles to max cycles" << endl
			<< "-F arg\t\tmax n. frames " << endl
			<< "-o arg\t\toutput filename (<filename>." << OUTFILE_EXT << ")"
			<< endl << "-p arg\t\tparallel configuration filename" << endl;
}

//parsing
void get_arguments(char *argv[], int argc, arguments &args) {
	args.alfa = ALFA_DEFAULT;
	args.beta = BETA_DEFAULT;
	args.w_max = MAX_WINDOW_SIZE;
	args.max_cycles = MAX_CYCLES_DEFAULT;
	args.fixed_cycles = false;
	args.verbose = false;
	args.user_out_fname = false;
	args.show_enabled = false;
	args.nframes = NFRAMES_DEFAULT;
	args.noise = 0;
	args.noise_type = SPNOISE;

#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && defined(_WIN32)
	TCHAR *options = "a:b:w:c:fo:n:svF:hN:";
#else
	const char *options = "a:b:w:c:fo:n:svF:hN:p:";
#endif
	int opt;

#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && defined(_WIN32)
	int opterr = 0;
#else
	opterr = 0;
#endif

	bool conf_fname = false;
	while ((opt = getopt(argc, argv, options)) != -1) {
		switch (opt) {
		case 'a': //alpha
			args.alfa = (float) atof(optarg);
			if (err_alfa(args.alfa)) {
				cerr << "ERROR in argument: a" << endl;
				print_help(argv[0]);
				exit(1);
			}
			break;
		case 'b': //beta
			args.beta = (float) atof(optarg);
			if (err_beta(args.beta)) {
				cerr << "ERROR in argument: b" << endl;
				print_help(argv[0]);
				exit(1);
			}
			break;
		case 'w': //window size
			args.w_max = atoi(optarg);
			if (err_w(args.w_max)) {
				cerr << "ERROR in argument: w" << endl;
				print_help(argv[0]);
				exit(1);
			}
			break;
		case 'c': //cycles max
			args.max_cycles = atoi(optarg);
			if (err_max_cycles(args.max_cycles)) {
				cerr << "ERROR in argument: c" << endl;
				print_help(argv[0]);
				exit(1);
			}
			break;
		case 'F': //n. frames max
			args.nframes = atoi(optarg);
			if (err_nframes(args.nframes)) {
				cerr << "ERROR in argument: F" << endl;
				print_help(argv[0]);
				exit(1);
			}
			break;
		case 'f': //fixed cycles
			args.fixed_cycles = true;
			break;
		case 'o': //output fname
			args.user_out_fname = true;
			args.out_fname.append(optarg); //file
			break;
		case 'p': //output fname
			conf_fname = true;
			args.conf_fname.append(optarg); //file
			break;
		case 'n': //noise-%
			args.add_noise = true;
			args.noise = atoi(optarg);
			if (err_noise(args.noise)) {
				cerr << "ERROR: noise-% not valid" << endl;
				exit(1);
			}
			break;
		case 'h': //help
			print_help(argv[0]);
			exit(0);
		case 'v': //verbose
			args.verbose = true;
			break;
		case 's': //show
			args.show_enabled = true;
			break;
		case 'N': //noise type
			args.noise_type = atoi(optarg);
			if (err_max_cycles(args.noise_type)) {
				cerr << "ERROR in argument: c" << endl;
				print_help(argv[0]);
				exit(1);
			}
			break;
		case '?': //parsing error
			cerr << "Illegal options" << endl;
			print_help(argv[0]);
			exit(1);
		}
	}
	//parse non-optional arguments
	if (optind < argc) {
		args.fname.append(argv[optind]); //file
		// if(err_bmp(args.fname)) {
		//   cerr << "ERROR: file not valid" << endl;
		//   exit(1);
		// }
	} else {
		// NO FILE = camera
		args.fname.append("VideoFromCamera.avi");
	}
	if (!conf_fname)
		cout << "WARNING: no parallel conf. file provided" << endl;
}

void get_parallel_parameters(arguments &args,
		parallel_parameters_t *parameters) {
	std::ifstream infile(args.conf_fname.c_str());
	string instring;
	while (infile >> instring) {
		if (instring == "farm_workers")
			infile >> parameters->n_farm_workers;
		else if (instring == "detector_workers")
			infile >> parameters->n_detector_workers;
		else if (instring == "denoiser_workers")
			infile >> parameters->n_denoiser_workers;
		else
			cerr << "unknown parallel parameter: " << instring << endl;
	}
}
