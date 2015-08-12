/*
 * main.cpp
 *
 *  Created on: Jun 1, 2015
 *      Author: droccom
 */

#define NROWS 10
#define NITERS 20
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

void print(unsigned char *bmp, unsigned long w, const char *label) {
	printf("+++ %s\n", label);
	//std::cout << "* " << label << std::endl;
	for (unsigned long r = 0; r < w; ++r) {
		printf("* ");
		for (unsigned long c = 0; c < w; ++c)
			printf("%c ", pixel2char(bmp[at(r, c, w)]));
		printf("\n");
	}
}

void print_diff(unsigned char *a, unsigned char *b, unsigned long w) {
	unsigned long ndiff = 0;
	for (unsigned long r = 0; r < w; ++r)
		for (unsigned long c = 0; c < w; ++c)
			if (a[at(r, c, w)] != b[at(r, c, w)]) {
				printf("a(%lu,%lu) = %c, b(%lu,%lu) = %c\n", r, c,
						pixel2char(a[at(r, c, w)]), r, c,
						pixel2char(b[at(r, c, w)]));
				++ndiff;
			}
	printf("+ %lu diff found\n", ndiff);
}

int main(int argc, char * argv[]) {
	printf("This is gameoflife\n");
	printf("syntax: %s nrows niter rngseed\n", argv[0]);

	//set parameters
	unsigned long nrows = (argc > 1) ? atoi(argv[1]) : NROWS;
	unsigned int niters = (argc > 2) ? atoi(argv[2]) : NITERS;
	unsigned int rngseed = (argc > 3) ? atoi(argv[3]) : 0;
	printf("using parameters: nrows = %lu, niter = %u, rngseed = %u\n", nrows,
			niters, rngseed);

	struct timeval tv1, tv2;

	//seed
	unsigned long length = nrows * nrows;
	unsigned char *bitmap_seed = (unsigned char *) malloc(
			length * sizeof(unsigned char));

//  //random
//	srand(rngseed);
//	for (unsigned long i = 0; i < length; ++i)
//		bitmap_seed[i] = (rand() % 100 < SEEDALIVE);

	//custom: ALIANTE
	for (unsigned long i = 0; i < length; ++i)
		bitmap_seed[i] = 0;
	bitmap_seed[at(5, 1, nrows)] = 1;
	bitmap_seed[at(5, 2, nrows)] = 1;
	bitmap_seed[at(5, 3, nrows)] = 1;
	bitmap_seed[at(4, 3, nrows)] = 1;
	bitmap_seed[at(3, 2, nrows)] = 1;
#ifdef VERIFY
	double sum = 0;
	for (unsigned long i = 0; i < length; ++i)
	sum += (double) bitmap_seed[i];
	sum /= length;
	printf("+ seed alive-%% = %f\n", sum * 100);
#endif

	//PAR run
	unsigned char *bitmap_out = (unsigned char *) malloc(
			length * sizeof(unsigned char));
	memcpy(bitmap_out, bitmap_seed, length * sizeof(unsigned char));
	oclTask oclt(bitmap_seed, bitmap_out, length, niters, nrows);
	ff::ff_stencilReduceLoopOCL_1D<oclTask> oclStencilReduceOneShot(oclt, mapf,
			reducef, 0, NULL, NACC, nrows + 1);
    //oclStencilReduceOneShot.pickGPU(1);
	gettimeofday(&tv1, NULL);
	oclStencilReduceOneShot.run_and_wait_end();
	gettimeofday(&tv2, NULL);
	double extime_par = ((double) tv2.tv_sec - (double) tv1.tv_sec) * 1000
			+ ((double) tv2.tv_usec - (double) tv1.tv_usec) / 1000;
	printf("par ex. time = %f ms\n", extime_par);
#ifdef PRINT
	print(bitmap_out, nrows, "SLR OUTPUT");
#endif

#ifdef VERIFY
	//SEQ run
#define swap(a,b) unsigned char *tmp = a; a = b; b = tmp;
	unsigned char *bitmap_in_seq = (unsigned char *) malloc(
			length * sizeof(unsigned char));
	memcpy(bitmap_in_seq, bitmap_seed, length * sizeof(unsigned char));
	unsigned char *bitmap_out_seq = (unsigned char *) malloc(
			length * sizeof(unsigned char));
	memcpy(bitmap_out_seq, bitmap_in_seq, length * sizeof(unsigned char));
	unsigned int w = nrows;
#ifdef PRINT
	print(bitmap_in_seq, nrows, "SEED");
#endif

    //std::cerr << "N of cores " << ff_numCores() << "\n";
    
	ParallelFor pf(FF_AUTO, true);

	gettimeofday(&tv1, NULL);
	swap(bitmap_in_seq, bitmap_out_seq);
	for (unsigned int iter = 0; iter < niters; ++iter) {
		swap(bitmap_in_seq, bitmap_out_seq);

		//for (unsigned long r = 0; r < nrows; ++r) {
		pf.parallel_for(0, nrows, [&](const long r) {
		  for (unsigned long c = 0; c < nrows; ++c) {
		    unsigned char alive_in = bitmap_in_seq[at(r, c, w)];
		    unsigned char naliven = 0;
		    naliven +=
		      has_top(r) && has_left(c) ? bitmap_in_seq[at(r - 1, c-1, w)] : 0;
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
	print_diff(bitmap_out, bitmap_out_seq, nrows);
	free(bitmap_out_seq);
	//printf("seq ex. time = %f ms\n", extime_seq);
	//printf("par vs seq speedup = %f\n", extime_seq / extime_par);
	printf("parfor ex. time = %f ms\n", extime_seq);
	printf("par vs parfor speedup = %f\n", extime_seq / extime_par);

#endif

	free(bitmap_out);
	free(bitmap_seed);
	return 0;
}
