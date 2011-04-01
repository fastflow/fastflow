/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         kmeans.h   (an OpenMP version)                            */
/*   Description:  header file for a simple k-means clustering program       */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _H_KMEANS
#define _H_KMEANS

#include "definitions.h"
#include <assert.h>

double** omp_kmeans(int, double**, int, int, int, double, int*);
double** seq_kmeans(double**, int, int, int, double, int*, int *, double*);

double** file_read(int, char*, int*, int*);
int     file_write(char*, int, int, int, double**, int*);


double  wtime(void);

extern int _debug;

#endif
