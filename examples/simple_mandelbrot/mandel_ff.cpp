/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \link
 *  \file mandel_ff.cpp
 *  \ingroup application_level
 *
 *  \brief 
 *
 */

/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/* 

   Author: Marco Aldinucci.   
   email:  aldinuc@di.unito.it
   date :  15/11/09
          
   Very simple streaming mandelbrot with pthread

*/

#include <unistd.h>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/spin-lock.hpp>
#include <ff/mapping_utils.hpp>
#include "marX2.h"
#include <math.h>


#define DIM 800
#define ITERATION 1024

using namespace ff;
static lock_t lock;

typedef struct outitem {
  unsigned char * M;
  int line;
} ostream_t;

const double range=3.0;
const double init_a=-2.125,init_b=-1.5;

double step = range/((double) DIM);
int dim = DIM;
int niter = ITERATION;

class Worker: public ff_node {
public:
  void * svc(void * task) {
	int * t = (int *)task;
	//std::cout << "Worker " << ff_node::get_my_id() 
	// << " received task " << *t << "\n";
	ostream_t * oi = (ostream_t*)malloc(sizeof(ostream_t));
	oi->M = (unsigned char *) malloc(dim*sizeof(char));
	// this is just a cut&paste of the sequential algorithim 
	{
	  // this is just alpha renamining
	  int i = 	oi->line = *t;
	  //
	  int j,k;
	  double im,a,b,a2,b2,cr;
		  
	  im=init_b+(step*i);
	  for (j=0;j<dim;j++)   {         
		a=cr=init_a+step*j;
		b=im;
		k=0;
		for (k=0;k<niter;k++)
		  {
			a2=a*a;
			b2=b*b;
			if ((a2+b2)>4.0) break;
			b=2*a*b+im;
			a=a2-b2+cr;
		  }
		oi->M[j] =  (unsigned char) 255-((k*255/niter)); 
		//printf("%3d ",	oi->M[j] );
	  } 
	  //printf("\n");
	}
	
	return oi;

  }
};

class Worker2: public ff_node {
public:
      void * svc(void * task) {
	int * t = (int *)task;
	//std::cout << "Worker " << ff_node::get_my_id() 
	// << " received task " << *t << "\n";
	ostream_t * oi = (ostream_t*)malloc(sizeof(ostream_t));
	oi->M = (unsigned char *) malloc(dim*sizeof(char));
	// this is just a cut&paste of the sequential algorithim 
	{
	  // this is just alpha renamining
	  int i = 	oi->line = *t;
	  //
	  int j,k;
	  double im,a,b,a2,b2,cr;
		  
	  im=init_b+(step*i);
	  for (j=0;j<dim;j++)   {         
		a=cr=init_a+step*j;
		b=im;
		k=0;
		for (k=0;k<niter;k++)
		  {
			a2=a*a;
			b2=b*b;
			if ((a2+b2)>4.0) break;
			b=2*a*b+im;
			a=a2-b2+cr;
		  }
		oi->M[j] =  (unsigned char) 255-((k*255/niter)); 
		//printf("%3d ",	oi->M[j] );
	  } 
	  //printf("\n");
	}
	

#if !defined(NO_DISPLAY)
	spin_lock(lock);
	ShowLine(oi->M,dim,oi->line); 
	spin_unlock(lock);
#endif
    	free(oi->M);
	free(oi);
	return GO_ON;
  }
};


// the gatherer filter
class Collector: public ff_node {
public:
  void * svc(void * task) {	  
	ostream_t * t = (ostream_t *)task;
	
	// std::cout << "Collector [" << t->line << "]\n";
#if !defined(NO_DISPLAY)
	ShowLine(t->M,dim,t->line); 
#endif
	//if (*t == -1) return NULL;
	free(t->M);
	free(t);
	return GO_ON;
  }
private:
  int init;
};

// the load-balancer filter
class Emitter: public ff_node {
public:
  Emitter(int max_task):ntask(max_task) {};
  
  void * svc(void *) {	
	int * task = new int(dim-ntask);
	--ntask;
	if (ntask<0) return NULL;
	return task;
  }
private:
  int ntask;
};



int main(int argc, char ** argv) {
  int ncores = ff_numCores();
  int r,retries=1,workers=2;
  double avg=0, var, * runs;

  if (argc<5) {
	printf("Usage: mandel_seq size niterations retries nworkers [0|1]\n\n\n");
  }
  else {
	dim = atoi(argv[1]);
	niter = atoi(argv[2]);
	step = range/((double) dim);
	retries = atoi(argv[3]);
	workers = atoi(argv[4]);
	if (argc==6) {
	    if (atoi(argv[5])) ncores=99;// it forces to use template with collector
	    else ncores=2;               // it forces to use template without collector
	}	 
  }

  runs = (double *) malloc(retries*sizeof(double));

  printf("Mandebroot set from (%g+I %g) to (%g+I %g)\n",
		 init_a,init_b,init_a+range,init_b+range);
  printf("resolution %d pixel, Max. n. of iterations %d - Using %d workers\n",dim*dim,niter,workers);	

  if (ncores>=4) 
      printf("\nNOTE: using farm template WITH the collector module!\n\n");
  else
      printf("\nNOTE: using farm template WITHOUT the collector module!\n\n");

#if !defined(NO_DISPLAY)
  SetupXWindows(dim,dim,1,NULL,"FF Mandelbroot");
#endif  

  for (r=0;r<retries;r++) {

      ff_farm<> farm(false, dim);
	std::vector<ff_node *>w;
	for (int k=0;k<workers;k++)
	    w.push_back((ncores>=4)? ((ff_node*)new Worker) : ((ff_node*)new Worker2));

	farm.add_workers(w);
	
	Emitter E(dim);
	farm.add_emitter(&E);
	
	Collector C;
	if (ncores>=4)
	    farm.add_collector(&C);
	
	if (farm.run_and_wait_end()<0) {
	  error("running farm\n");
	  return -1;
	}
	avg += runs[r] = farm.ffTime();
	for (int k=0;k<workers;k++)
	    delete (Worker*)(w[k]);

	std::cerr << "Run [" << r << "] DONE, time= " <<  runs[r] << " (ms)\n";
  }
  // stats
  avg = avg / (double) retries;
  var = 0;
  for (r=0;r<retries;r++) {
	var += (runs[r] - avg) * (runs[r] - avg);
  }
  var /= retries;
  std::cerr << "Average on " << retries << " experiments = " << avg << " (ms) Std. Dev. " << sqrt(var) << "\n\nPress a key\n" << std::endl;
  getchar();
	
#if !defined(NO_DISPLAY)
  CloseXWindows();
#endif
  return 0;
}

/*!
 *  @}
 *  \endlink
 */
