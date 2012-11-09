#include <cmath>
#include "filter.h"
#include "utils.h"
using namespace std;

//utilities
#define MIN(A,B) ((A)>(B))?((B)):((A))

void Savgol::ludcmp(Mat_IO_DP &a, Vec_O_INT &indx, DP &d)
{
  const DP TINY=1.0e-20;
  int i,imax = 0,j,k;
  DP big,dum,sum,temp;

  int n=a.nrows();
  Vec_DP vv(n);
  d=1.0;
  for (i=0;i<n;i++) {
    big=0.0;
    for (j=0;j<n;j++)
      if ((temp=fabs(a[i][j])) > big) big=temp;
    if (big == 0.0) filter_error("Singular matrix in routine ludcmp");
    vv[i]=1.0/big;
  }
  for (j=0;j<n;j++) {
    for (i=0;i<j;i++) {
      sum=a[i][j];
      for (k=0;k<i;k++) sum -= a[i][k]*a[k][j];
      a[i][j]=sum;
    }
    big=0.0;
    for (i=j;i<n;i++) {
      sum=a[i][j];
      for (k=0;k<j;k++) sum -= a[i][k]*a[k][j];
      a[i][j]=sum;
      if ((dum=vv[i]*fabs(sum)) >= big) {
	big=dum;
	imax=i;
      }
    }
    if (j != imax) {
      for (k=0;k<n;k++) {
	dum=a[imax][k];
	a[imax][k]=a[j][k];
	a[j][k]=dum;
      }
      d = -d;
      vv[imax]=vv[j];
    }
    indx[j]=imax;
    if (a[j][j] == 0.0) a[j][j]=TINY;
    if (j != n-1) {
      dum=1.0/(a[j][j]);
      for (i=j+1;i<n;i++) a[i][j] *= dum;
    }
  }
}

void Savgol::lubksb(Mat_I_DP &a, Vec_I_INT &indx, Vec_IO_DP &b)
{
  int i,ii=0,ip,j;
  DP sum;

  int n=a.nrows();
  for (i=0;i<n;i++) {
    ip=indx[i];
    sum=b[ip];
    b[ip]=b[i];
    if (ii != 0)
      for (j=ii-1;j<i;j++) sum -= a[i][j]*b[j];
    else if (sum != 0.0)
      ii=i+1;
    b[i]=sum;
  }
  for (i=n-1;i>=0;i--) {
    sum=b[i];
    for (j=i+1;j<n;j++) sum -= a[i][j]*b[j];
    b[i]=sum/a[i][i];
  }
}

void Savgol::savgol(Vec_O_DP &c, const int np, const int nl, const int nr,
		const int ld, const int m)
{
  int j,k,imj,ipj,kk,mm;
  DP d,fac,sum;

  if (np < nl+nr+1 || nl < 0 || nr < 0 || ld > m || nl+nr < m) {
    filter_error("bad args in savgol");
    exit(-1);
  }
  Vec_INT indx(m+1);
  Mat_DP a(m+1,m+1);
  Vec_DP b(m+1);
  for (ipj=0;ipj<=(m << 1);ipj++) {
    sum=(ipj ? 0.0 : 1.0);
    for (k=1;k<=nr;k++) sum += pow(DP(k),DP(ipj));
    for (k=1;k<=nl;k++) sum += pow(DP(-k),DP(ipj));
    mm=MIN(ipj,2*m-ipj);
    for (imj = -mm;imj<=mm;imj+=2) a[(ipj+imj)/2][(ipj-imj)/2]=sum;
  }
  ludcmp(a,indx,d);
  for (j=0;j<m+1;j++) b[j]=0.0;
  b[ld]=1.0;
  lubksb(a,indx,b);
  for (kk=0;kk<np;kk++) c[kk]=0.0;
  for (k = -nl;k<=nr;k++) {
    sum=b[0];
    fac=1.0;
    for (mm=1;mm<=m;mm++) sum += b[mm]*(fac *= k);
    kk=(np-k) % np;
    c[kk]=sum;
  }
}

Savgol_Prediction::Savgol_Prediction(const int nl, const int nr, const int m, DP time_step, DP time_win ) {
  c.resize(nl+nr+1);
  cp.resize(nl+nr+1);
  c1.resize(nl+nr+1);
  c2.resize(nl+nr+1);
  Savgol::savgol(c,nl+nr+1,nl,nr,0,m);
  Savgol::savgol(c1,nl+nr+1,nl,nr,1,m);
  Savgol::savgol(c2,nl+nr+1,nl,nr,2,m);
  cp=(c1+2.*c2*time_step/time_win)*time_step/time_win;  
  c1/=time_win;
  c2/=(time_win*time_win);
  c2*=2.;
}

/*
  double operator()(Vec_I_DP& x, Vec_I_DP& y) {	
  return sqrt(pow(c*(x-y),2.).sum());
  }
*/

double Savgol_Prediction::prediction(Vec_I_DP& x) {
  return ((c+cp)*x).sum();
}

double Savgol_Prediction::filt(Vec_I_DP& x) {
  return (c*x).sum();
}

bool Savgol_Prediction::peak(Vec_I_DP& x, DP f1, DP f2) {
  /*
  if (fabs((c1*x).sum()) < f1 && (c2*x).sum() < -f2) return true;
  else return false;
  */
  //cout <<fabs((c1*x).sum())<<"<->"<< (c2*x).sum()<< " ";
  return fabs((c1*x).sum()) < f1 && (c2*x).sum() < -f2; //il segno meno alla derivata seconda non dovrebbe esserci... Non mi so spiegare il motivo
}
