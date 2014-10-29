/* ***********************************************************************
  This program is part of the
        OpenMP Source Code Repository

        http://www.pcg.ull.es/ompscr/
        e-mail: ompscr@etsii.ull.es

   Copyright (c) 2004, OmpSCR Group
   All rights reserved.

   Redistribution and use in source and binary forms, with or without modification, 
   are permitted provided that the following conditions are met:
     * Redistributions of source code must retain the above copyright notice, 
       this list of conditions and the following disclaimer. 
     * Redistributions in binary form must reproduce the above copyright notice, 
       this list of conditions and the following disclaimer in the documentation 
       and/or other materials provided with the distribution. 
     * Neither the name of the University of La Laguna nor the names of its contributors 
       may be used to endorse or promote products derived from this software without 
       specific prior written permission. 

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
   IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
   OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
   OF SUCH DAMAGE.

  FILE:              c_jacobi01.c (THIS IS THE ORIGINAL FILE NAME)
  VERSION:           1.1
  DATE:              Oct 2004
  AUTHORS:           Author:       Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998
                     Modified:     Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998
                     This version: Dieter an Mey,     Aachen University (RWTH), 1999 - 2003
                                   anmey@rz.rwth-aachen.de
                                   http://www.rwth-aachen.de/People/D.an.Mey.html
  COMMENTS TO:       ompscr@etsii.ull.es
  DESCRIPTION:       program to solve a finite difference discretization of Helmholtz equation : 
                     (d2/dx2)u + (d2/dy2)u - alpha u = f using Jacobi iterative method.
  COMMENTS:          OpenMP version 1: two parallel regions with one parallel loop each, the naive approach.  
                     Directives are used in this code to achieve paralleism. 
                     All do loops are parallized with default 'static' scheduling.
  REFERENCES:        http://www.rz.rwth-aachen.de/computing/hpc/prog/par/openmp/jacobi.html
  BASIC PRAGMAS:     parallel for
  USAGE:             ./c_jacobi01.par 5000 5000 0.8 1.0 1000
  INPUT:             n - grid dimension in x direction
                     m - grid dimension in y direction
                     alpha - Helmholtz constant (always greater than 0.0)
                     tol   - error tolerance for iterative solver
                     relax - Successice over relaxation parameter
                     mits  - Maximum iterations for iterative solver
  OUTPUT:            Residual and error 
                     u(n,m) - Dependent variable (solutions)
                     f(n,m) - Right hand side function 
  FILE FORMATS:      -
  RESTRICTIONS:      -
  REVISION HISTORY:
**************************************************************************/

/* Minor modifications (mainly for auto vectorization of loops) 
 * to the original code and added the FastFlow and TBB parallel loops code
 *   
 *  Author: Massimo Torquati <torquati@di.unipi.it>
 *
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include <ff/utils.hpp>
#if defined(USE_TBB) 
#include <numeric>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#endif

#include <ff/parallel_for.hpp>

using namespace ff;

#define U(i,j) u[(i)*n+(j)]
#define F(i,j) f[(i)*n+(j)]
#define NUM_ARGS  6
#define NUM_TIMERS 1

int n, m, mits;
double tol, relax, alpha;
int NUMTHREADS;

static long chunk=1;


/* 
      subroutine jacobi (n,m,dx,dy,alpha,omega,u,f,tol,maxit)
******************************************************************
* Subroutine HelmholtzJ
* Solves poisson equation on rectangular grid assuming : 
* (1) Uniform discretization in each direction, and 
* (2) Dirichlect boundary conditions 
* 
* Jacobi method is used in this routine 
*
* Input : n,m   Number of grid points in the X/Y directions 
*         dx,dy Grid spacing in the X/Y directions 
*         alpha Helmholtz eqn. coefficient 
*         omega Relaxation factor 
*         f(n,m) Right hand side function 
*         u(n,m) Dependent variable/Solution
*         tol    Tolerance for iterative solver 
*         maxit  Maximum number of iterations 
*
* Output : u(n,m) - Solution 
*****************************************************************
*/
#if defined(USE_OPENMP)
static inline void jacobi_omp ( const int n, const int m, const double dx, const double dy, const double alpha, 
				const double omega, double * __restrict__ u, double * __restrict__ f, const double tol, const int maxit )
{
  int k;
  double Error, resid;

  double *__restrict__ uold;

  /* wegen Array-Kompatibilitaet, werden die Zeilen und Spalten (im Kopf)
	 getauscht, zB uold[spalten_num][zeilen_num]; bzw. wir tuen so, als ob wir das
	 gespiegelte Problem loesen wollen */

  uold = (double *)malloc(sizeof(double) * n *m);

  const double ax = 1.0/(dx * dx); /* X-direction coef */
  const double ay = 1.0/(dy*dy); /* Y_direction coef */
  const double b = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */

  Error = 10.0 * tol;

  k = 1;
  while (k <= maxit && Error > tol) {

	Error = 0.0;

	//unsigned long t1 = getusec();
	/* copy new solution into old */
#pragma omp parallel for schedule(runtime) num_threads(NUMTHREADS)
	for (int j=0; j<m; j++) {
	  double * __restrict__ _uold=uold;
	  double * __restrict__ _u   =u;
	  const int mj = m*j;	  
          PRAGMA_IVDEP;
	  for (int i=0; i<n; i++)
	    _uold[i + mj] = _u[i + mj];
	}
	//unsigned long t2 = getusec();
	/* compute stencil, residual and update */
#pragma omp parallel for reduction(+:Error) schedule(runtime) private(resid) num_threads(NUMTHREADS)
	for (int j=1; j<m-1; j++) {
	  double * __restrict__ _uold=uold;
	  double * __restrict__ _u   =u;
	  double * __restrict__ _f   =f;				    

	  const int  _n = n-1;
	  const int mj = m*j;
	  const int mjp1 = m*(j+1);
	  const int mjm1 = m*(j-1);

	  PRAGMA_IVDEP;
	  for (int i=1; i<_n; i++){
	    resid =(
		    ax * (_uold[i-1 + mj] + _uold[i+1 + mj])
		    + ay * (_uold[i + mjm1] + _uold[i + mjp1])
		    + b * _uold[i + mj] - _f[i + mj]
		    ) / b;
		
	    /* update solution */
	    _u[i + mj] = _uold[i + mj] - omega * resid;
	    
	    /* accumulate residual error */
	    Error =Error + resid*resid;
	    
	  }
	}
	//unsigned long t3 = getusec();
	//printf("%d %g  %g\n", k, (double)((t2-t1))/1000.0, (double)((t3-t2))/1000.0);
	/* error check */
	k++;
	Error = sqrt(Error) /(n*m);
  } /* while */

  printf("Total Number of Iterations %d\n", k-1);
  printf("Residual                   %.15f\n\n", Error);

  free(uold);
} 	

#elif defined(USE_TBB)

static inline void jacobi_tbb ( const int n, const int m, const double dx, const double dy, const double alpha, 
				const double omega, double * __restrict__ u, double * __restrict__ f, const double tol, const int maxit )
{
  int k;
  double Error;

  tbb::affinity_partitioner ap;

  double *__restrict__ uold;

  /* wegen Array-Kompatibilitaet, werden die Zeilen und Spalten (im Kopf)
     getauscht, zB uold[spalten_num][zeilen_num]; bzw. wir tuen so, als ob wir das
     gespiegelte Problem loesen wollen */
  
  uold = (double *)malloc(sizeof(double) * n *m);

  const double ax = 1.0/(dx * dx); /* X-direction coef */
  const double ay = 1.0/(dy*dy); /* Y_direction coef */
  const double b = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */

  auto Fcopy = [&uold,&u,n,m] (const tbb::blocked_range<long>& r) {
    double * __restrict__ _uold=uold;
    double * __restrict__ _u   =u;
    
    for (long j=r.begin();j!=r.end();++j) {
      const long _n = n;
      const long mj = m*j;
      PRAGMA_IVDEP;
      for (long i=0; i<_n; i++)
	_uold[i + mj] = _u[i + mj];
    }
  };
  auto Freduce = [&uold,&u,f,n,m,ax,ay,b,omega](const tbb::blocked_range<long> &r, double in) -> double {
    double * __restrict__ _uold=uold;
    double * __restrict__ _u   =u;
    double * __restrict__ _f   =f;				    
    
    for (long j=r.begin();j!=r.end();++j) {
      const long _n = n-1;
      const long mj = m*j;
      const long mjp1 = m*(j+1);
      const long mjm1 = m*(j-1);
      
      PRAGMA_IVDEP;
      for (long i=1; i<_n; i++){
	const double resid =(
			     ax * (_uold[i-1 + mj] + _uold[i+1 + mj])
			     + ay * (_uold[i + mjm1] + _uold[i + mjp1])
			     + b * _uold[i + mj] - _f[i + mj]
			     ) / b;
	
	/* update solution */
	_u[i + mj] = _uold[i + mj] - omega * resid;
	in+=resid*resid;
      }
    }
    return in;
  };

  Error = 10.0 * tol;

  k = 1;
  while (k <= maxit && Error > tol) {
    
    Error = 0.0;

    //unsigned long t1 = getusec();
    /* copy new solution into old */
    tbb::parallel_for(tbb::blocked_range<long>(0, m), Fcopy, ap);
    //unsigned long t2 = getusec();
    /* compute stencil, residual and update */
    Error += tbb::parallel_reduce(tbb::blocked_range<long>(1, m-1),double(0),
				  Freduce, std::plus<double>(), ap );
    //unsigned long t3 = getusec();
    //printf("%d %g  %g\n", k, (double)((t2-t1))/1000.0, (double)((t3-t2))/1000.0);
    /* error check */
    k++;
    Error = sqrt(Error) /(n*m);
    
  } /* while */

  printf("Total Number of Iterations %d\n", k-1);
  printf("Residual                   %.15f\n\n", Error);

  free(uold);
} 	
#else  // FF

static inline void jacobi_ff (ParallelForReduce<double> &pfr, 
			      const int n, const int m, const double dx, const double dy, const double alpha, 
			      double omega, double * __restrict__ u, double * __restrict__ f, const double tol, const int maxit )
{
  int k;
  double Error;


  double * __restrict__ uold;

  /* wegen Array-Kompatibilitaet, werden die Zeilen und Spalten (im Kopf)
	 getauscht, zB uold[spalten_num][zeilen_num]; bzw. wir tuen so, als ob wir das
	 gespiegelte Problem loesen wollen */

  uold = (double *)malloc(sizeof(double) * n *m);

  const double ax = 1.0/(dx * dx); /* X-direction coef */
  const double ay = 1.0/(dy*dy); /* Y_direction coef */
  const double b = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */

  Error = 10.0 * tol;

  auto Fsum = [](double& v, const double elem) { v += elem; };
  auto Fcopy = [&uold,&u,n,m](const long j) {
    double * __restrict__ _uold=uold;
    double * __restrict__ _u   =u;
    const int _n = n;
    const int mj = m*j;
    PRAGMA_IVDEP;
    for (int i=0; i<_n; i++)
      _uold[i + mj] = _u[i + mj];
  };
  auto Freduce = [&uold,&u,f,n,m,ax,ay,b,omega](const long j, double& Error) {
    double * __restrict__ _uold=uold;
    double * __restrict__ _u   =u;
    double * __restrict__ _f   =f;
    
    const long _n   = n-1;
    const long mj   = m*j;
    const long mjp1 = mj+m;
    const long mjm1 = mj-m;
    
    PRAGMA_IVDEP;
    for (long i=1; i<_n; i++){
      const double resid =(
			   ax * (_uold[i-1 + mj] + _uold[i+1 + mj])
			   + ay * (_uold[i + mjm1] + _uold[i + mjp1])
			   + b * _uold[i + mj] - _f[i + mj]
			   ) / b;
      
      /* update solution */
      _u[i + mj] = _uold[i + mj] - omega * resid;
      
      /* accumulate residual error */
      Error += resid*resid;	      
    }
  };

  k = 1;
  while (k <= maxit && Error > tol) {
    Error = 0.0;
    //unsigned long t1 = getusec();
    /* copy new solution into old */
    pfr.parallel_for(0,m,1, chunk, Fcopy,NUMTHREADS);
    //unsigned long t2 = getusec();

    /* compute stencil, residual and update */
    pfr.parallel_reduce(Error, 0.0, 1,m-1,1, chunk,Freduce,Fsum,NUMTHREADS);

    //unsigned long t3 = getusec();
    //printf("%d %g  %g\n", k, (double)((t2-t1))/1000.0, (double)((t3-t2))/1000.0);    
    /* error check */
    k++;
    Error = sqrt(Error) /(n*m);
  } /* while */

  printf("Total Number of Iterations %d\n", k-1);
  printf("Residual                   %.15f\n\n", Error);

  free(uold);  
} 	
#endif


/******************************************************
* Initializes data 
* Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
*
******************************************************/
void initialize(  
                int n,    
                int m,
                double alpha,
                double *dx,
                double *dy,
                double *u,
                double *f)
{
  int xx,yy;

  *dx = 2.0 / (n-1);
  *dy = 2.0 / (m-1);

  /* Initilize initial condition and RHS */
  for (int j=0; j<m; j++){
    for (int i=0; i<n; i++){
      xx = -1.0 + *dx * (i-1);
      yy = -1.0 + *dy * (j-1);
      U(j,i) = 0.0;
      F(j,i) = -alpha * (1.0 - xx*xx) * (1.0 - yy*yy)
                - 2.0 * (1.0 - xx*xx) - 2.0 * (1.0 - yy*yy);
    }
  }
      
}


/************************************************************
* Checks error between numerical and exact solution 
*
************************************************************/
void error_check(
                 int n,
                 int m,
                 double alpha,
                 double dx,
                 double dy,
                 double *u,
                 double *f)
{
  double xx, yy, temp, error;

  dx = 2.0 / (n-1);
  dy = 2.0 / (n-2);
  error = 0.0;

  for (int j=0; j<m; j++){
    for (int i=0; i<n; i++){
      xx = -1.0 + dx * (i-1);
      yy = -1.0 + dy * (j-1);
      temp = U(j,i) - (1.0 - xx*xx) * (1.0 - yy*yy);
      error += temp*temp;
    }
  }

 error = sqrt(error)/(n*m);

  printf("Solution Error : %g\n", error);

}




int main(int argc, char **argv){
    double *u, *f, dx, dy;
    double dt, mflops;

    if (argc<8) {
	printf("use:%s n m alpha relax tot mits nthreadds\n", argv[0]);
	printf(" example %s 5000 5000 0.8 1.0 1e-7 1000 4\n",argv[0]);
	return -1;
    }

   n = atoi(argv[1]);
   m = atoi(argv[2]);
   alpha = atof(argv[3]);
   relax = atof(argv[4]);
   tol = atof(argv[5]);
   mits = atoi(argv[6]);
   NUMTHREADS = atoi(argv[7]);
   if (argc==9) chunk = atoi(argv[8]);

   //printf("-> %d, %d, %g, %g, %g, %d\n",
   //	    n, m, alpha, relax, tol, mits);
   //printf("-> NUMTHREADS=%d\n", NUMTHREADS);

   u = (double *)malloc(n*m*sizeof(double));
   f = (double *)malloc(n*m*sizeof(double));

   initialize(n, m, alpha, &dx, &dy, u, f);

#if defined(USE_OPENMP)
   /* OpenMP */
   
   // warm-up
   jacobi_omp(n, m, dx, dy, alpha, relax, u,f, tol, 1);


   printf("OpenMP runs using %d threads\n", NUMTHREADS);
   ffTime(START_TIME);
   /* Solve Helmholtz equation */
   jacobi_omp(n, m, dx, dy, alpha, relax, u,f, tol, mits);
   ffTime(STOP_TIME);
   dt = ffTime(GET_TIME); 

   printf("omp elapsed time : %12.6f  (ms)\n", dt);
#elif defined(USE_TBB)
   tbb::task_scheduler_init init(NUMTHREADS);

   // warm-up
   jacobi_tbb(n, m, dx, dy, alpha, relax, u,f, tol, 1);

   printf("TBB runs using %d threads\n", NUMTHREADS);
   ffTime(START_TIME);
   /* Solve Helmholtz equation */
   jacobi_tbb(n, m, dx, dy, alpha, relax, u,f, tol, mits);
   ffTime(STOP_TIME);
   dt = ffTime(GET_TIME); 

   printf("TBB elapsed time : %12.6f  (ms)\n", dt);
#else
   /* FastFlow */

#if defined(__MIC__)
    const char worker_mapping[]="1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 0, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 130, 134, 138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186, 190, 194, 198, 202, 206, 210, 214, 218, 222, 226, 230, 234, 237, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 238, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 239";
    threadMapper::instance()->setMappingList(worker_mapping);
#endif

    ParallelForReduce<double> pfr(NUMTHREADS, (NUMTHREADS < ff_numCores()));

    // warm-up
    jacobi_ff(pfr, n, m, dx, dy, alpha, relax, u,f, tol, 1);

    printf("\n\nFastFlow runs using %d threads\n", NUMTHREADS);
    ffTime(START_TIME);
    /* Solve Helmholtz equation */
    jacobi_ff(pfr, n, m, dx, dy, alpha, relax, u,f, tol, mits);
    ffTime(STOP_TIME);
    dt = ffTime(GET_TIME); 
    printf("ff elapsed time : %12.6f (ms)\n", dt);

#endif

   mflops = (0.000001*mits*(m-2)*(n-2)*13) / (dt/1000.0);
   printf(" MFlops       : %12.6g (%d, %d, %d, %g)\n",mflops, mits, m, n, (dt/1000.0));

   error_check(n, m, alpha, dx, dy, u, f);
   
   return 0;
}

