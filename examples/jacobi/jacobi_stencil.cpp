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
#include <ff/stencil.hpp>

using namespace ff;

int NUMTHREADS;
long CHUNKSIZE=-1;


stencil2D<double> *stencil=NULL;

#define U(i,j) u[(i)*n+(j)]
#define F(i,j) f[(i)*n+(j)]
#define NUM_ARGS  6
#define NUM_TIMERS 1

int n, m, mits;
double tol, relax, alpha;



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

static inline void jacobi( const int n, const int m, double dx, double dy, double alpha, 
			   double omega, double *u, double *f, double tol, int maxit )
{
  double ax, ay, b;

  /* wegen Array-Kompatibilitaet, werden die Zeilen und Spalten (im Kopf)
	 getauscht, zB uold[spalten_num][zeilen_num]; bzw. wir tuen so, als ob wir das
	 gespiegelte Problem loesen wollen */

  ax = 1.0/(dx * dx); /* X-direction coef */
  ay = 1.0/(dy*dy); /* Y_direction coef */
  b = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */

  // initialize uold
  auto initUold = [u] (ff_forall_farm<double> * loopInit, double *M, size_t Xsize, size_t Ysize) {
    FF_PARFOR_START(loopInit, j, 0, Xsize, 1, CHUNKSIZE, NUMTHREADS) {
      for(size_t i=0;i<Xsize;++i) {
	M[i + Xsize*j] = u[i+Xsize*j];
      }
    } FF_PARFOR_STOP(loopInit);
  };
  // pre-computation function
  auto reset = [](double *, const long, const long,double& Error) { Error = 0; };

  struct {
      double ax,ay,b,omega;
      double *f;
  } p = {ax,ay,b,omega,f};
  // stencil function
  auto stencilF = [&p](ff_forall_farm<double> *loopCompute, double *in, double *out, 
		       const size_t Xsize, const size_t Xstart, const size_t Xstop,
		       const size_t Ysize, const size_t Ystart, const size_t Ystop, 
		       double &Error) {
    double Err = Error;  // to avoid compiler warning
    FF_PARFORREDUCE_START(loopCompute,Err,0.0,j,1,Xstop,1, CHUNKSIZE, loopCompute->getNWorkers()) {
      double resid;
      for(size_t i=1;i<Ystop;++i) {
	resid = (p.ax * (in[i-1 + Xsize*j] + in[i+1 + Xsize*j])
		 + p.ay * (in[i + Xsize*(j-1)] + in[i + Xsize*(j+1)])
		 + p.b * in[i + Xsize*j] - p.f[i + Xsize*j]
		 ) / p.b;	
	
	out[i+Xsize*j] = (in[i+ Xsize*j] - p.omega*resid);
	
	Err = Err + resid*resid;
      }
    } FF_PARFORREDUCE_STOP(loopCompute,Err,+);
    Error = Err;
  };
    
  // reduce operation
  auto reduceOp = [](double& E, double V) { E += V; };

  // condition function
  struct {
    double tol;
    const int n,m;
  } p2 = {tol,n,m};

  auto condF = [&p2](double E, const size_t) -> bool { 
    return ( (sqrt(E)/(p2.m*p2.n)) > p2.tol); 
  };

  stencil->initOutFuncAll(initUold);
  stencil->preFunc(reset);
  stencil->computeFuncAll(stencilF, 1,m-1,1, 1,n-1,1);
  stencil->reduceFunc(condF, maxit, reduceOp, 0.0);
  stencil->run_and_wait_end();

  printf("Total Number of Iterations %ld\n", stencil->getIter());  
  printf("Residual                   %.15f\n\n", (sqrt(stencil->getReduceVar())/(m*n)));

} 	




int main(int argc, char **argv){
    double *u, *f, dx, dy;
    double dt, mflops;

    if (argc<9) {
	printf("use:%s n m alpha relax tot mits nthreadds chunksize\n", argv[0]);
	printf(" example %s 5000 5000 0.8 1.0 1e-7 1000 4 1000\n",argv[0]);
	return -1;
    }

   n = atoi(argv[1]);
   m = atoi(argv[2]);
   alpha = atof(argv[3]);
   relax = atof(argv[4]);
   tol = atof(argv[5]);
   mits = atoi(argv[6]);
   NUMTHREADS = atoi(argv[7]);
   CHUNKSIZE  = atoi(argv[8]);

   //printf("-> %d, %d, %g, %g, %g, %d\n",
   //	    n, m, alpha, relax, tol, mits);
   //printf("-> NUMTHREADS=%d\n", NUMTHREADS);

   u = (double *) malloc(n*m*sizeof(double));
   f = (double *) malloc(n*m*sizeof(double));

   initialize(n, m, alpha, &dx, &dy, u, f);
   double *uold = (double *)malloc(sizeof(double) * n *m);

#if !defined(FUNCTIONS)
   stencil = new stencil2D<double>(u,uold,m,n,n,NUMTHREADS,1,1,false,CHUNKSIZE);
#endif

   ffTime(START_TIME);
   /* Solve Helmholtz equation */
#if defined(FUNCTIONS)
   jacobi_functions(n, m, dx, dy, alpha, relax, u,f, tol, mits);
#else
   jacobi(n, m, dx, dy, alpha, relax, u,f, tol, mits);
#endif
   ffTime(STOP_TIME);
   dt = ffTime(GET_TIME); 

   printf("elapsed time : %12.6f  (ms)\n", dt);


   mflops = (0.000001*mits*(m-2)*(n-2)*13) / (dt/1000.0);
   printf(" MFlops       : %12.6g (%d, %d, %d, %g)\n",mflops, mits, m, n, (dt/1000.0));

   error_check(n, m, alpha, dx, dy, u, f);
   free(uold);
   return 0;
}

