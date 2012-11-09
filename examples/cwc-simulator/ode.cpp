/*
  This file is part of CWC Simulator.

  CWC Simulator is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  CWC Simulator is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with CWC Simulator.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "ode.h"
#include "Species.h"
#include <gsl/gsl_errno.h>
#include <my_gsl_odeiv.h>
#include <iostream>
#include <cmath>
using namespace std;

int func (double t, const double x[], double dxdt[], void *params)
{
  double *k = (double *)params; //rates
  /* TAT *//*
     dxdt[0] = - k[10] * x[0]; //CMRNA
     dxdt[1] = k[2] * x[0] - k[9] * x[1]; //CTAT
     dxdt[2] = k[1] * x[0] - k[8] * x[2] ; //GFP
     dxdt[3] = k[7] * x[7] - k[3] * x[5] * x[3] + k[4] * x[6]; //LTR
     dxdt[4] = k[7] * x[7] - k[11] * x[4]; //NMRNA
     dxdt[5] = - k[3] * x[5] * x[3] + k[4] * x[6] + k[7] * x[7]; //TAT
     dxdt[6] = k[3] * x[5] * x[3] - k[4] * x[6] - k[5] * x[6] + k[6] * x[7]; //TEF
     dxdt[7] = k[5] * x[6] - k[6] * x[7] - k[7] * x[7]; //TEFa*/

  /* Competitive LK
  //     k={1 1 1 0.0015 0.0015 0.0015 0.002 0.0015 0.002 1 1 1 0.0015 0.0015 0.0015 0.002 0.0015 0.002}
  dxdt[0] = k[0] * x[0] - k[3] * x[0] * x[0] - k[6] * x[0] * x[1] - k[7] * x[0] * x[2] ; //A
  dxdt[1] = k[1] * x[1] - k[4] * x[1] * x[1] - k[6] * x[0] * x[1] - k[8] * x[1] * x[2]; //B
  dxdt[2] = k[2] * x[2] - k[5] * x[2] * x[2] - k[7] * x[0] * x[2] - k[8] * x[1] * x[2]; //C
  dxdt[3] = k[9] * x[3] - k[12] * x[3] * x[3] - k[15] * x[3] * x[4] - k[16] * x[3] * x[5] ; //IN A
  dxdt[4] = k[10] * x[4] - k[13] * x[4] * x[4] - k[15] * x[3] * x[4] - k[17] * x[4] * x[5]; //IN B
  dxdt[5] = k[11] * x[5] - k[14] * x[5] * x[5] - k[16] * x[3] * x[5] - k[17] * x[4] * x[5]; //IN Ci */
  
  /* TAT (without compartments) *//*
     dxdt[0] = - k[1] * x[3] - k[11] * x[0]; //CmRNA
     dxdt[1] = k[2] * x[0] - k[9] * x[1] ; //GFP
     dxdt[2] = k[4] * x[4] * x[2] + k[5] * x[6] + k[8] * x[5]; //LTR
     dxdt[3] = k[0] * x[2] - k[1] * x[3] + k[8] * x[5] - k[12] * x[3]; //NmRNA
     dxdt[4] = k[3] * x[5] - k[4] * x[4] * x[2] + k[5] * x[6] + k[8] * x[5] - k[10] * x[4]; //TATd
     dxdt[5] = k[6] * x[6] - k[7] * x[5] - k[8] * x[5]; //TEFa
     dxdt[6] = k[4] * x[4] * x[2] - k[5] * x[6] - k[6] * x[6] + k[7] * x[5]; //TEFd*/

  /* QS
  dxdt[0] = k[1] * x[4] + k[8] * x[5] -k[9] * x[0]; //LasI
  dxdt[1] = k[0] * x[4] - k[4] * x[2] * x[1] + k[7] * x[5] -k[10] * x[1]; //LasR
  dxdt[2] = k[3] * x[0]+ k[5] * x[6] - k[4] * x[2] * x[1] - k[11] * x[2]; //oxo3
  dxdt[3] = k[4] * x[2] * x[1] - k[1] * x[3] - k[5] * x[3] * x[4] + k[6] * x[5]; //R3
  dxdt[4] = k[3] * x[5] - k[4] * x[4] * x[2] - k[5] * x[3] * x[4] + k[6] * x[5]; //LasO.LasR.LasI
  dxdt[5] = k[5] * x[3] * x[4] - k[6] * x[5] ; //RO3.LasR.LasI
  dxdt[6] = -k[12] * x[6]; // oxo3out*/

  /* lkvir
     dxdt[0] = k[0] * x[0] - k[1] * x[0] * x[1] - k[3]*x[0]*x[2] ; //A
     dxdt[1] = -k[2] * x[1] + k[1] * x[0] * x[1] ; //B
     dxdt[2] = 0; //vir*/
  
  
  /* virus
     dxdt[0] = - k[0] * x[0] + k[2] * x[2] -k[3] * x[0] * x[1]; //gen
     dxdt[1] = k[2] * x[0] - k[3] * x[0] * x[1] + k[4] * x[2] - k[5] * x[1]; //struct
     dxdt[2] = k[0] * x[0] - k[1] * x[2] ; //tem */
  
/* compartmentalized virus */
     dxdt[0] = - k[0] * x[0] + k[2] * x[2]; //gen
     dxdt[1] = k[2] * x[0] + k[3] * x[2] - k[4] * x[1]; //struct
     dxdt[2] = k[0] * x[0] - k[1] * x[2] ; //tem */
  

 /* crist
     dxdt[0] = - k[0] * x[0] * x[0] -k[1] * x[0] * x[2]; //a
     dxdt[1] = (k[0]/2.0) * x[0] * x[0]; //b
     dxdt[2] = - k[1] * x[0] * x[2]; //c
     dxdt[3] = k[1] * x[0] * x[2]; //d */
 
  return GSL_SUCCESS;
}

void odeSolver(double ** data, double *k, double t0, double tf, double delta, unsigned int n_species)
{
  const gsl_odeiv_step_type * T 
    = gsl_odeiv_step_rk8pd;

  gsl_odeiv_step * s 
    = gsl_odeiv_step_alloc (T, n_species);
  gsl_odeiv_control * c 
    = gsl_odeiv_control_y_new (1e-6, 0.0);
  gsl_odeiv_evolve * e 
    = gsl_odeiv_evolve_alloc (n_species);

  
  //double p[7] ={111.289066, 0.060545, 23.048264, 26.500049, 109.754423, 199.804324, 0.001582};
  gsl_odeiv_system sys = {func, NULL, n_species, k};

  double t = t0, t1;
  double h = delta/3.0;
  
  double y[n_species];
  for(unsigned int i=0; i<n_species; i++) y[i]=data[0][i];
  int i;
  for (i=1, t1=delta;t1<tf;t1+=delta, i++){
    while (t < t1)
      {
	int status = gsl_odeiv_evolve_apply (e, c, s,
					     &sys, 
					     &t, t1,
					     &h, y);

	if (status != GSL_SUCCESS) break; 
      }
    for(unsigned int j=0; j<n_species; j++) data[i][j]=y[j];
    //    printf ("%.5e %.5e %.5e %.5e\n", tf, y[0], y[1],y[2]);
  }
     if(t<tf){
       /*int status = */gsl_odeiv_evolve_apply (e, c, s,
                                             &sys,
                                             &t, tf,
                                             &h, y);

       //if (status != GSL_SUCCESS) break;
    for(unsigned int j=0; j<n_species; j++) data[i][j]=y[j];
    }

  gsl_odeiv_evolve_free (e);
  gsl_odeiv_control_free (c);
  gsl_odeiv_step_free (s);
}
