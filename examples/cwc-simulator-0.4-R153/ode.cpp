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
#include <gsl/gsl_errno.h>
#include <my_gsl_odeiv.h>
#include <iostream>
#include <cmath>
using namespace std;

#define NPar 6 //number of species
#define NPoint 2

int func (double t, const double x[], double dxdt[], void *params)
{
  double *k = (double *)params; //rates

  /* Competitive LK */
  // k={1 1 1 0.0015 0.0015 0.0015 0.002 0.0015 0.002 1 1 1 0.0015 0.0015 0.0015 0.002 0.0015 0.002}
  dxdt[0] = k[0] * x[0] - 2 * k[3] * x[0] * x[0] - k[6] * x[0] * x[1] - k[7] * x[0] * x[2] ; //A
  dxdt[1] = k[1] * x[1] - 2 * k[4] * x[1] * x[1] - k[6] * x[0] * x[1] - k[8] * x[1] * x[2]; //B
  dxdt[2] = k[2] * x[2] - 2 * k[5] * x[2] * x[2] - k[7] * x[0] * x[2] - k[8] * x[1] * x[2]; //C
  dxdt[3] = k[9] * x[3] - 2* k[12] * x[3] * x[3] - k[15] * x[3] * x[4] - k[16] * x[3] * x[5] ; //IN A
  dxdt[4] = k[10] * x[4] - 2 * k[13] * x[4] * x[4] - k[15] * x[3] * x[4] - k[17] * x[4] * x[5]; //IN B
  dxdt[5] = k[11] * x[5] - 2 * k[14] * x[5] * x[5] - k[16] * x[3] * x[5] - k[17] * x[4] * x[5]; //IN C

  /* TAT (without compartments) *//*
  dxdt[0] = - k[1] * x[3] - k[11] * x[0]; //CmRNA
  dxdt[1] = k[2] * x[0] - k[9] * x[1] ; //GFP
  dxdt[2] = k[4] * x[4] * x[2] + k[5] * x[6] + k[8] * x[5]; //LTR
  dxdt[3] = k[0] * x[2] - k[1] * x[3] + k[8] * x[5] - k[12] * x[3]; //NmRNA
  dxdt[4] = k[3] * x[5] - k[4] * x[4] * x[2] + k[5] * x[6] + k[8] * x[5] - k[10] * x[4]; //TATd
  dxdt[5] = k[6] * x[6] - k[7] * x[5] - k[8] * x[5]; //TEFa
  dxdt[6] = k[4] * x[4] * x[2] - k[5] * x[6] - k[6] * x[6] + k[7] * x[5]; //TEFd*/

  return GSL_SUCCESS;
}

void odeSolver(double ** data, double *k, double t0, double tf, double delta)
{
  const gsl_odeiv_step_type * T 
    = gsl_odeiv_step_rk8pd;

  gsl_odeiv_step * s 
    = gsl_odeiv_step_alloc (T, NPar);
  gsl_odeiv_control * c 
    = gsl_odeiv_control_y_new (1e-6, 0.0);
  gsl_odeiv_evolve * e 
    = gsl_odeiv_evolve_alloc (NPar);

  
  //double p[7] ={111.289066, 0.060545, 23.048264, 26.500049, 109.754423, 199.804324, 0.001582};
  gsl_odeiv_system sys = {func, NULL, NPar, k};

  double t = t0, t1;
  double h = 0.05;
  
  double y[NPar];
  for(int i=0; i<NPar; i++) y[i]=data[0][i];
  int i;
  for (i=1, t1=delta;t1<=tf;t1+=delta, i++){
	  while (t < t1)
	    {
	      int status = gsl_odeiv_evolve_apply (e, c, s,
						   &sys, 
						   &t, t1,
						   &h, y);

	      if (status != GSL_SUCCESS) break; 
	  }
	  for(int j=0; j<NPar; j++) data[i][j]=y[j];
	  //printf ("%.5e %.5e %.5e %.5e\n", t, y[0], y[1],y[2]);
  }

  gsl_odeiv_evolve_free (e);
  gsl_odeiv_control_free (c);
  gsl_odeiv_step_free (s);
}

void limited_ode(vector<double> &k, Species &x, double tau) {
  vector<multiplicityType> &subject(x.concentrations);

#ifdef DEBUG_SIMULATION
  cerr << "(tau: " << tau << ") input ODE: ";
  for(unsigned int i=0; i<subject.size(); i++)
    cerr << subject[i] << " ";
  cerr << endl;
#endif

  //cast rates to array
  int nk(k.size());
  double *k_array = (double *)malloc(nk * sizeof(double));
  for(int i=0; i<nk; i++)
    k_array[i] = k[i];

#ifdef DEBUG_SIMULATION
  cerr << "rates: ";
  for(int i=0; i<nk; i++)
    cerr << k_array[i] << " ";
  cerr << endl;
#endif

  //get concentrations (from multiplicities)
  double **data = (double **)malloc(NPoint * sizeof(double *));
  for(int i=0; i<NPoint; i++)
    data[i] = (double *)malloc(NPar * sizeof(double));
  for(int i=0; i<NPar; i++)
    data[0][i] = double(subject[i]);

  odeSolver(data, k_array, 0.0, tau, tau);

  //set multiplicities (from concentrations)
  for (int i=0; i < NPar; i++)
    subject[i] = multiplicityType(round(data[NPoint-1][i])); //last state

  //clean
  free(k_array);
  for(int i=0; i<NPoint; i++)
    free(data[i]);
  free(data);

#ifdef DEBUG_SIMULATION
  cerr << "output ODE: ";
  for(unsigned int i=0; i<subject.size(); i++)
    cerr << subject[i] << " ";
  cerr << endl;
#endif
}
