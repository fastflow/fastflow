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

#if !defined(FF_CUDA)
#define FF_CUDA
#endif

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include <ff/utils.hpp>
#include <ff/stencilReduceCUDA.hpp>

using namespace ff;

#define U(i,j) u[(i)*n+(j)]
#define F(i,j) f[(i)*n+(j)]

struct const_t {
    const_t(int n, int m, float ax, float ay, float b, float omega):
	n(n),m(m),ax(ax),ay(ay),b(b),omega(omega) {}

    int n,m;
    float ax, ay, b,omega;
};

// name, outT,inT,in,env1T,env1,env2T,env2,env3T,env3,env4T,env4
FFMAPFUNC4(mapF, float, int, idx, float, U, float, UOLD, float, F, const_t, CONST, 
	   const int    n    = CONST->n;
	   const float ax    = CONST->ax;
	   const float ay    = CONST->ay;
	   const float b     = CONST->b;
	   const float omega = CONST->omega;	  	  
	   const int left    = idx - 1;
	   const int right   = idx + 1;
	   const int up      = idx - n;
	   const int down    = idx + n;
	   
	   float resid = (
	  		  ax * (UOLD[left] + UOLD[right])
	  		  + ay * (UOLD[down] + UOLD[up])
	  		  + b * UOLD[idx] - F[idx]
			  ) / b;
	   
	   U[idx] = UOLD[idx] - omega * resid;
	   return resid*resid;
);

FFREDUCEFUNC(reduceF, float, x, y, 
	     return x+y;
);

    
class jacobiCUDATask: public baseCUDATask<int, float, float, float, float, const_t> {
public:		      
  jacobiCUDATask(int n, int m, float dx, float dy, float alpha, float omega,
		 float *u, float *f, float tol):n(n),m(m),dx(dx),dy(dy),
						alpha(alpha),omega(omega),u(u),f(f),
						tol(tol),Error(0.0),resid(0.0) { 
    
    
    const float ax = 1.0/(dx * dx); /* X-direction coef */
    const float ay = 1.0/(dy*dy); /* Y_direction coef */
    const float b = -2.0/(dx*dx)-2.0/(dy*dy) - alpha; /* Central coeff */
    
    
    Const   = new const_t(n,m,ax,ay,b,omega);     // TODO: free
    Indexes = new int[(n-2)*(m-2)];               // TODO: free
    Residual = new float[(n-2)*(m-2)];            // TODO: free
    uold     = (float*)malloc(m*n*sizeof(float));// TODO: free
    memcpy(uold, u, m*n*sizeof(float));
    
    int k=0;
    for (int j=1; j<m-1; j++)
      for (int i=1; i<n-1; i++) {
	Indexes[k] = i+m*j;
	Residual[k] = 0.0;
	k++;
      }
    
  }
  jacobiCUDATask():m(-1),n(-1),dx(0.0),dy(0.0),
		   alpha(0.0),omega(0.0),u(NULL),f(NULL),
		   tol(0.0),Error(0.0),resid(0.0) { }
  

    jacobiCUDATask & operator=(const jacobiCUDATask &t) { 
	n=t.n; m=t.m; dx=t.dx; dy=t.dy; alpha=t.alpha; omega=t.omega;
	u=t.u; f=t.f; tol=t.tol; Error=t.Error; resid=t.resid;
	Indexes=t.Indexes;
	Residual=t.Residual;
	uold=t.uold;
	Const=t.Const;
        return *this; 
    }

    void cleanMemory() {
      if (Const) delete Const; 
      if (Indexes) delete [] Indexes;
      if (Residual) delete [] Residual;
      if (uold) free(uold);
      Const = NULL, Indexes=NULL, Residual=NULL, uold=NULL;
    }	 
    

    void setTask(void* t) {
	const jacobiCUDATask &task = *(jacobiCUDATask*)t;

	this->operator=(task);

	setInPtr(Indexes);
	setSizeIn((n-2)*(m-2));
	setOutPtr(Residual);
	setSizeOut((n-2)*(m-2));
	
	setEnv1Ptr(u);
	setSizeEnv1(m*n);
	setEnv2Ptr(uold);
	setSizeEnv2(m*n);       
	setEnv3Ptr(f);
	setSizeEnv3(m*n);
	setEnv4Ptr(Const);
	setSizeEnv4(1);
    }
    
    void beforeMR() { setReduceVar(0.0); }

    bool iterCondition(float E, size_t iter) { 
	Iter  = iter;
	Error = sqrt(E) / (n*m);
	return ( Error > tol);  
    }

    void swap() {
	float *tmp = getEnv1DevicePtr();       // U
	setEnv1DevicePtr(getEnv2DevicePtr());  // U=UOLD
	setEnv2DevicePtr(tmp);                 // UOLD=U
    }    
    
    void endMR(void *t) {
        jacobiCUDATask &task = *(jacobiCUDATask*)t;


        task.setResid(getReduceVar());
	task.setError(Error);    
	task.setIter(Iter);

	task.cleanMemory();
    }

    float getError() const { return Error;}
    float getResid() const { return resid;}
    size_t getIter() const { return Iter; }
    void   setError(float e) { Error = e;}
    void   setResid(float r) { resid = r;}
    void   setIter(size_t i) { Iter  = i;}
private:
    int n, m;
    float dx, dy, alpha, omega;
    float *u;
    float *f;
    float tol;
    float Error;
    float resid;
    size_t Iter;

protected:
    const_t *Const;
    int     *Indexes;
    float   *Residual;
    float   *uold;
};



/******************************************************
* Initializes data 
* Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
*
******************************************************/
void initialize(  
                int n,    
                int m,
                float alpha,
                float *dx,
                float *dy,
                float *u,
                float *f)
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
                 float alpha,
                 float dx,
                 float dy,
                 float *u,
                 float *f)
{
  float xx, yy, temp, error;

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
    float *u, *f, dx, dy;
    float dt, mflops;
    int n, m, mits;
    float tol, relax, alpha;

    if (argc<7) {
	printf("use:%s n m alpha relax tot mits\n", argv[0]);
	printf(" example %s 5000 5000 0.8 1.0 1e-7 1000\n",argv[0]);
	return -1;
    }

   n = atoi(argv[1]);
   m = atoi(argv[2]);
   alpha = atof(argv[3]);
   relax = atof(argv[4]);
   tol = atof(argv[5]);
   mits = atoi(argv[6]);

   u = (float *)malloc(n*m*sizeof(float));
   f = (float *)malloc(n*m*sizeof(float));

   initialize(n, m, alpha, &dx, &dy, u, f);


   jacobiCUDATask jt(n, m, dx, dy, alpha, relax,u,f,tol);
   FFSTENCILREDUCECUDA(jacobiCUDATask, mapF, reduceF) jacobi(jt, mits);

   printf("Jacobi started\n");
   ffTime(START_TIME);
   /* Solve Helmholtz equation */
   jacobi.run_and_wait_end();
   ffTime(STOP_TIME);
   dt = ffTime(GET_TIME); 

   printf("Total Number of Iterations %d\n", (int)jt.getIter());
   printf("Residual                   %.15f\n\n", jt.getError());

   printf("elapsed time : %12.6f  (ms)\n", dt);

   mflops = (0.000001*mits*(m-2)*(n-2)*13) / (dt/1000.0);
   printf(" MFlops       : %12.6g (%d, %d, %d, %g)\n",mflops, mits, m, n, (dt/1000.0));

   error_check(n, m, alpha, dx, dy, u, f);
   
   return 0;
}

