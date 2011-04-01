#define NPar 8 //number of species

int func (double t, const double x[], double dxdt[], void *params)
{
  double *k = (double *)params; //rates

  /* TAT*/
  dxdt[0] = - k[10] * x[0]; //CMRNA
  dxdt[1] = k[2] * x[0] - k[9] * x[1]; //CTAT
  dxdt[2] = k[1] * x[0] - k[8] * x[2] ; //GFP
  dxdt[3] = k[7] * x[7] - k[3] * x[5] * x[3] + k[4] * x[6]; //LTR
  dxdt[4] = k[7] * x[7] - k[11] * x[4]; //NMRNA
  dxdt[5] = - k[3] * x[5] * x[3] + k[4] * x[6] + k[7] * x[7]; //TAT
  dxdt[6] = k[3] * x[5] * x[3] - k[4] * x[6] - k[5] * x[6] + k[6] * x[7]; //TEF
  dxdt[7] = k[5] * x[6] - k[6] * x[7] - k[7] * x[7]; //TEFa

  return GSL_SUCCESS;
}
