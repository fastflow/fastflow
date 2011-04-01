#define NPar 7 //number of species

int func (double t, const double x[], double dxdt[], void *params)
{
  double *k = (double *)params; //rates

  /* TAT (without compartments) */
  dxdt[0] = - k[1] * x[3] - k[11] * x[0]; //CmRNA
  dxdt[1] = k[2] * x[0] - k[9] * x[1] ; //GFP
  dxdt[2] = k[4] * x[4] * x[2] + k[5] * x[6] + k[8] * x[5]; //LTR
  dxdt[3] = k[0] * x[2] - k[1] * x[3] + k[8] * x[5] - k[12] * x[3]; //NmRNA
  dxdt[4] = k[3] * x[5] - k[4] * x[4] * x[2] + k[5] * x[6] + k[8] * x[5] - k[10] * x[4]; //TATd
  dxdt[5] = k[6] * x[6] - k[7] * x[5] - k[8] * x[5]; //TEFa
  dxdt[6] = k[4] * x[4] * x[2] - k[5] * x[6] - k[6] * x[6] + k[7] * x[5]; //TEFd

  return GSL_SUCCESS;
}
