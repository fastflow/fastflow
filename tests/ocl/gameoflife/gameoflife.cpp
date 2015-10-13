/*
 * main.cpp
 *
 *  Created on: Jun 1, 2015
 *      Author: droccom
 */

#define NROWS 1024
#define NITERS 100
#define SEEDALIVE 30 //%

#include <cstdio>
#include <iostream>
#include <string>
#include <sys/time.h>

#include <ff/parallel_for.hpp>
#include "gameoflifeSRL.hpp"

using namespace ff;

#define CHECK 1
#ifdef CHECK
#if !defined(VERIFY)
#define VERIFY
#endif
#include "../ctest.h"
#else
#define NACC 1
#endif


#define at(r,c,w) (r)*w+c
#define has_top(r) (r > 0)
#define has_left(c) (c > 0)
#define has_right(c,w) (c < w-1)
#define has_bottom(r,w) (r < w-1)

inline char pixel2char(unsigned char x) {
	return x ? '*' : ' ';
}

void print(unsigned char *bmp, unsigned int w, const char *label) {
	printf("+++ %s\n", label);
	//std::cout << "* " << label << std::endl;
	for (unsigned int r = 0; r < w; ++r) {
		printf("* ");
		for (unsigned int c = 0; c < w; ++c)
			printf("%c ", pixel2char(bmp[at(r, c, w)]));
		printf("\n");
	}
}

unsigned int get_and_print_diff(unsigned char *a, unsigned char *b, unsigned int w) {
	unsigned int ndiff = 0;
	for (unsigned int r = 0; r < w; ++r)
		for (unsigned int c = 0; c < w; ++c)
			if (a[at(r, c, w)] != b[at(r, c, w)]) {
				printf("a(%u,%u) = %c, b(%u,%u) = %c\n", r, c,
						pixel2char(a[at(r, c, w)]), r, c,
						pixel2char(b[at(r, c, w)]));
				++ndiff;
			}
	printf("+ %u diff found\n", ndiff);
	return ndiff;
}

int main(int argc, char * argv[]) {
	printf("This is gameoflife\n");
	printf("syntax: %s nrows niter rngseed\n", argv[0]);

	//set parameters
	unsigned int nrows = (argc > 1) ? atoi(argv[1]) : NROWS;
	unsigned int niters = (argc > 2) ? atoi(argv[2]) : NITERS;
	unsigned int rngseed = (argc > 3) ? atoi(argv[3]) : 0;
	printf("using parameters: nrows = %u, niter = %u, rngseed = %u\n", nrows,
			niters, rngseed);

	struct timeval tv1, tv2;

	//seed
	unsigned int length = nrows * nrows;
	unsigned char *bitmap_seed = (unsigned char *) malloc(
			length * sizeof(unsigned char));

  //random
	srand(rngseed);
	for (unsigned int i = 0; i < length; ++i)
		bitmap_seed[i] = (rand() % 100 < SEEDALIVE);

//	//custom: ALIANTE
//	for (unsigned int i = 0; i < length; ++i)
//		bitmap_seed[i] = 0;
//	bitmap_seed[at(3, 2, nrows)] = 1;
//	bitmap_seed[at(4, 3, nrows)] = 1;
//	bitmap_seed[at(5, 1, nrows)] = 1;
//	bitmap_seed[at(5, 2, nrows)] = 1;
//	bitmap_seed[at(5, 3, nrows)] = 1;
#ifdef VERIFY
	double sum = 0;
	for (unsigned int i = 0; i < length; ++i)
	sum += (double) bitmap_seed[i];
	sum /= length;
	printf("+ seed alive-%% = %f\n", sum * 100);
#endif
#ifdef PRINT
	print(bitmap_seed, nrows, "SEED");
#endif

	//OpenCL run
	unsigned char *bitmap_out = (unsigned char *) malloc(
			length * sizeof(unsigned char));
	memcpy(bitmap_out, bitmap_seed, length * sizeof(unsigned char));
	oclTaskGol oclt(bitmap_seed, bitmap_out, niters, nrows);
	//create a 3-by-3 2D stencilReduceLoop node
#if defined(BUILD_WITH_SOURCE)
	ff::ff_stencilReduceLoopOCL_2D<oclTaskGol> oclStencilReduceOneShot(oclt, 
									   "cl_code/gameoflife.cl",
									   "mapf",
									   "reducef", 0, NULL, NACC, 1, 1);

	oclStencilReduceOneShot.saveBinaryFile(); 	
	oclStencilReduceOneShot.reuseBinaryFile();
#else
	ff::ff_stencilReduceLoopOCL_2D<oclTaskGol> oclStencilReduceOneShot(oclt, 
									   mapf,
									   reducef, 0, NULL, NACC, 1, 1);
#endif

	SET_DEVICE_TYPE(oclStencilReduceOneShot);
	gettimeofday(&tv1, NULL);
	oclStencilReduceOneShot.run_and_wait_end();
	gettimeofday(&tv2, NULL);
	double extime_par = ((double) tv2.tv_sec - (double) tv1.tv_sec) * 1000
			+ ((double) tv2.tv_usec - (double) tv1.tv_usec) / 1000;
	printf("OCL ex. time = %f ms\n", extime_par);
#ifdef PRINT
	print(bitmap_out, nrows, "OCL OUTPUT");
#endif

#ifdef VERIFY
	//multicore run
#define swap(a,b) unsigned char *tmp = a; a = b; b = tmp;
	unsigned char *bitmap_in_seq = (unsigned char *) malloc(
			length * sizeof(unsigned char));
	memcpy(bitmap_in_seq, bitmap_seed, length * sizeof(unsigned char));
	unsigned char *bitmap_out_seq = (unsigned char *) malloc(
			length * sizeof(unsigned char));
	memcpy(bitmap_out_seq, bitmap_in_seq, length * sizeof(unsigned char));
	unsigned int w = nrows;

    //std::cerr << "N of cores " << ff_numCores() << "\n";
    
	ParallelFor pf(FF_AUTO, true);

	gettimeofday(&tv1, NULL);
	swap(bitmap_in_seq, bitmap_out_seq);
	for (unsigned int iter = 0; iter < niters; ++iter) {
		swap(bitmap_in_seq, bitmap_out_seq);

		//for (unsigned int r = 0; r < nrows; ++r) {
		pf.parallel_for(0, nrows, [&](const long r) {
		  for (unsigned int c = 0; c < nrows; ++c) {
		    unsigned char alive_in = bitmap_in_seq[at(r, c, w)];
		    unsigned char naliven = 0;
		    naliven +=
		      has_top(r) && has_left(c) ? bitmap_in_seq[at(r-1, c-1, w)] : 0;
		    naliven += has_top(r) ? bitmap_in_seq[at(r - 1, c, w)] : 0;
		    naliven +=
		      has_top(r) && has_right(c, w) ? bitmap_in_seq[at(r - 1, c+1, w)] : 0;
		    naliven += has_right(c,w) ? bitmap_in_seq[at(r, c+1, w)] : 0;
		    naliven +=
		      has_bottom(r,w) && has_right(c, w) ? bitmap_in_seq[at(r + 1, c+1, w)] : 0;
		    naliven += has_bottom(r,w) ? bitmap_in_seq[at(r + 1, c, w)] : 0;
		    naliven +=
		      has_bottom(r,w) && has_left(c) ? bitmap_in_seq[at(r + 1, c-1, w)] : 0;
		    naliven += has_left(c) ? bitmap_in_seq[at(r, c-1, w)] : 0;
		    bitmap_out_seq[at(r, c, w)] = naliven == 3 || (alive_in && naliven == 2);
		  }
		  }
		  );
	}
	gettimeofday(&tv2, NULL);
	double extime_seq = ((double)tv2.tv_sec - (double)tv1.tv_sec) * 1000 + ((double)tv2.tv_usec - (double)tv1.tv_usec) / 1000;
#ifdef PRINT
	print(bitmap_out_seq, nrows, "SEQ OUTPUT");
#endif
	int ndiff = get_and_print_diff(bitmap_out, bitmap_out_seq, nrows);
	free(bitmap_out_seq);
	free(bitmap_in_seq);
	//printf("seq ex. time = %f ms\n", extime_seq);
	//printf("par vs seq speedup = %f\n", extime_seq / extime_par);
	printf("parfor ex. time = %f ms\n", extime_seq);
	printf("par vs parfor speedup = %f\n", extime_seq / extime_par);

#endif

	free(bitmap_out);
	free(bitmap_seed);

#ifdef CHECK
	if (ndiff) {
			printf("Error\n");
			exit(1); //ctest
		}
#endif

	return 0;
}
