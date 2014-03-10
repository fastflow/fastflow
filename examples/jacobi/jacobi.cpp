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

/* Minor modifications to the original code and the FastFlow code done
 * by Massimo Torquati <torquati@di.unipi.it>
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

#if !defined(USE_OPENMP) && !defined(USE_TBB)
#include <ff/parallel_for.hpp>
#endif


using namespace ff;

#if defined(STATIC_DECL)
static FF_PARFOR_DECL(loop)=NULL;
static FF_PARFORREDUCE_DECL(reduce,double)=NULL;
#endif


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
static inline void jacobi_omp ( const int n, const int m, double dx, double dy, double alpha, 
	double omega, double *u, double *f, double tol, int maxit )
{
  int k;
  double Error, resid, ax, ay, b;


  double *uold;

  /* wegen Array-Kompatibilitaet, werden die Zeilen und Spalten (im Kopf)
	 getauscht, zB uold[spalten_num][zeilen_num]; bzw. wir tuen so, als ob wir das
	 gespiegelte Problem loesen wollen */

  uold = (double *)malloc(sizeof(double) * n *m);



  ax = 1.0/(dx * dx); /* X-direction coef */
  ay = 1.0/(dy*dy); /* Y_direction coef */
  b = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */

  Error = 10.0 * tol;

  k = 1;
  while (k <= maxit && Error > tol) {

	Error = 0.0;

	/* copy new solution into old */
#pragma omp parallel for schedule(runtime) num_threads(NUMTHREADS)
    for (int j=0; j<m; j++)
	  for (int i=0; i<n; i++)
		uold[i + m*j] = u[i + m*j];

	/* compute stencil, residual and update */
#pragma omp parallel for reduction(+:Error) schedule(runtime) private(resid) num_threads(NUMTHREADS)
	for (int j=1; j<m-1; j++)
	  for (int i=1; i<n-1; i++){
		resid =(
			ax * (uold[i-1 + m*j] + uold[i+1 + m*j])
			+ ay * (uold[i + m*(j-1)] + uold[i + m*(j+1)])
			+ b * uold[i + m*j] - f[i + m*j]
			) / b;
		
		/* update solution */
		u[i + m*j] = uold[i + m*j] - omega * resid;

		/* accumulate residual error */
		Error =Error + resid*resid;

	  }

	/* error check */
	k++;
	Error = sqrt(Error) /(n*m);
  } /* while */

  printf("Total Number of Iterations %d\n", k-1);
  printf("Residual                   %.15f\n\n", Error);

  free(uold);

} 	
#elif defined(USE_TBB)
static inline void jacobi_tbb ( const int n, const int m, double dx, double dy, double alpha, 
	double omega, double *u, double *f, double tol, int maxit )
{
  int k;
  double Error, resid, ax, ay, b;


  tbb::affinity_partitioner ap;

  double *uold;

  /* wegen Array-Kompatibilitaet, werden die Zeilen und Spalten (im Kopf)
	 getauscht, zB uold[spalten_num][zeilen_num]; bzw. wir tuen so, als ob wir das
	 gespiegelte Problem loesen wollen */

  uold = (double *)malloc(sizeof(double) * n *m);



  ax = 1.0/(dx * dx); /* X-direction coef */
  ay = 1.0/(dy*dy); /* Y_direction coef */
  b = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */

  Error = 10.0 * tol;

  k = 1;
  while (k <= maxit && Error > tol) {
    
    Error = 0.0;

    /* copy new solution into old */
    tbb::parallel_for(tbb::blocked_range<long>(0, m, m/NUMTHREADS),
		      [&] (const tbb::blocked_range<long>& r) {
			for (long j=r.begin();j!=r.end();++j) {
			  for (int i=0; i<n; i++)
			    uold[i + m*j] = u[i + m*j];
			}
		      }, ap);

    /* compute stencil, residual and update */
    Error += tbb::parallel_reduce(tbb::blocked_range<long>(1, m-1), //,(m-2)/NUMTHREADS),
    				  double(0),
    				  [&] (tbb::blocked_range<long> &r, double in) -> double {
    				    for (long j=r.begin();j!=r.end();++j) {
    				      for (int i=1; i<n-1; i++){
    				      	double resid =(
    				      		ax * (uold[i-1 + m*j] + uold[i+1 + m*j])
    				      		+ ay * (uold[i + m*(j-1)] + uold[i + m*(j+1)])
    				      		+ b * uold[i + m*j] - f[i + m*j]
    				      		) / b;
					
    				      	/* update solution */
    				      	u[i + m*j] = uold[i + m*j] - omega * resid;
    				      	in+=resid*resid;
    				      }
    				    }
    				    return in;
    				  }, std::plus<double>(), ap );
    
    /* error check */
    k++;
    Error = sqrt(Error) /(n*m);
    
  } /* while */

  printf("Total Number of Iterations %d\n", k-1);
  printf("Residual                   %.15f\n\n", Error);

  free(uold);

} 	
#else  // FF

static inline void jacobi_ff ( const int n, const int m, double dx, double dy, double alpha, 
	double omega, double *u, double *f, double tol, int maxit )
{
  int k;
  double Error, ax, ay, b;


  double *uold;

  /* wegen Array-Kompatibilitaet, werden die Zeilen und Spalten (im Kopf)
	 getauscht, zB uold[spalten_num][zeilen_num]; bzw. wir tuen so, als ob wir das
	 gespiegelte Problem loesen wollen */

  uold = (double *)malloc(sizeof(double) * n *m);



  ax = 1.0/(dx * dx); /* X-direction coef */
  ay = 1.0/(dy*dy); /* Y_direction coef */
  b = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */

  Error = 10.0 * tol;

#if !defined(STATIC_DECL)
  FF_PARFOR_INIT(loop, NUMTHREADS);
  FF_PARFORREDUCE_INIT(reduce, double, NUMTHREADS); 
#endif

  k = 1;
  while (k <= maxit && Error > tol) {

	Error = 0.0;

	/* copy new solution into old */
	FF_PARFOR_START(loop, j,0,m,1, chunk, NUMTHREADS) { //(m/NUMTHREADS),NUMTHREADS) {
	    for (int i=0; i<n; i++)
		uold[i + m*j] = u[i + m*j];
	} FF_PARFOR_STOP(loop);

	/* compute stencil, residual and update */
	FF_PARFORREDUCE_START(reduce, Error, 0.0, j,1,m-1,1, chunk, NUMTHREADS) { //(m-2)/NUMTHREADS, NUMTHREADS) { 
	    double resid;
	    for (int i=1; i<n-1; i++){
		resid =(
			ax * (uold[i-1 + m*j] + uold[i+1 + m*j])
			+ ay * (uold[i + m*(j-1)] + uold[i + m*(j+1)])
			+ b * uold[i + m*j] - f[i + m*j]
			) / b;
		
		/* update solution */
		u[i + m*j] = uold[i + m*j] - omega * resid;
		
		/* accumulate residual error */
		Error =Error + resid*resid;
		
	    }
	} FF_PARFORREDUCE_STOP(reduce, Error, +);

	/* error check */
	k++;
	Error = sqrt(Error) /(n*m);
  } /* while */

  printf("Total Number of Iterations %d\n", k-1);
  printf("Residual                   %.15f\n\n", Error);

  free(uold);  
#if !defined(STATIC_DECL)
  FF_PARFOR_DONE(loop);
  FF_PARFORREDUCE_DONE(reduce);
#endif
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

   u = (double *) malloc(n*m*sizeof(double));
   f = (double *) malloc(n*m*sizeof(double));

   initialize(n, m, alpha, &dx, &dy, u, f);

#if defined(USE_OPENMP)
   /* OpenMP */

   printf("OpenMP runs using %d threads\n", NUMTHREADS);
   ffTime(START_TIME);
   /* Solve Helmholtz equation */
   jacobi_omp(n, m, dx, dy, alpha, relax, u,f, tol, mits);
   ffTime(STOP_TIME);
   dt = ffTime(GET_TIME); 

   printf("omp elapsed time : %12.6f  (ms)\n", dt);
#elif defined(USE_TBB)
   tbb::task_scheduler_init init(NUMTHREADS);

   printf("TBB runs using %d threads\n", NUMTHREADS);
   ffTime(START_TIME);
   /* Solve Helmholtz equation */
   jacobi_tbb(n, m, dx, dy, alpha, relax, u,f, tol, mits);
   ffTime(STOP_TIME);
   dt = ffTime(GET_TIME); 

   printf("TBB elapsed time : %12.6f  (ms)\n", dt);
#else
   /* FastFlow */

#if defined(STATIC_DECL)
  FF_PARFOR_ASSIGN(loop, NUMTHREADS);
  FF_PARFORREDUCE_ASSIGN(reduce, double, NUMTHREADS); 
#endif

   printf("\n\nFastFlow runs using %d threads\n", NUMTHREADS);
   ffTime(START_TIME);
   /* Solve Helmholtz equation */
   jacobi_ff(n, m, dx, dy, alpha, relax, u,f, tol, mits);
   ffTime(STOP_TIME);
   dt = ffTime(GET_TIME); 

   printf("ff elapsed time : %12.6f (ms)\n", dt);

#if defined(STATIC_DECL)
   FF_PARFOR_DONE(loop);
   FF_PARFORREDUCE_DONE(reduce);
#endif
#endif

   mflops = (0.000001*mits*(m-2)*(n-2)*13) / (dt/1000.0);
   printf(" MFlops       : %12.6g (%d, %d, %d, %g)\n",mflops, mits, m, n, (dt/1000.0));

   error_check(n, m, alpha, dx, dy, u, f);
   
   return 0;
}

