#ifndef _SPD_CONVERGENCE_HPP
#define _SPD_CONVERGENCE_HPP
#include "utils.hpp"
#include "bitmap.hpp"
#include "control_structures.hpp"
#include <vector>
//#include <cstdlib>  // for abs
using namespace std;

typedef float residual_t;
typedef vector<residual_t> residual_set; //residual for a noisy-set

/*
  --------------
  sets residuals
  --------------
 */
template <typename T>
void partial_reduce_residual_max(Bitmap<T> &bmp,
				 vector<vector<noisy<T> > > &row,
				 vector<vector<T> > &diff,
				 vector<residual_t> &res,
				 long cluster_index
				 )
{
  long tmp, tmpd, max = 0;
  for (unsigned int i=0; i<row[cluster_index].size();i++) {
    tmp = _ABS((long)bmp.get(row[cluster_index][i].c,row[cluster_index][i].r) - (long)row[cluster_index][i].original_pixel);
    tmpd = _ABS(((tmp - (long)diff[cluster_index][i])));
    max = _MAX(max,tmpd);
    /*
    if(tmp == 255 || tmp == 0)
      cerr << "(" << row[cluster_index][i].c << "," << row[cluster_index][i].r << ") "
	   << "orig: " << (long)row[cluster_index][i].original_pixel
	   << ", new: "<< (long)bmp.get(row[cluster_index][i].c,row[cluster_index][i].r)
	   << ", D(p,i) = " << tmp << ", D(p, i-1) = " << (long)diff[cluster_index][i] << ", R(p,i) = " << tmpd
	   << " [maxR(set #" << cluster_index << ", i): " << max << "]" << endl;
    */
    diff[cluster_index][i] = (T)tmp;
  }
  res[cluster_index] = (residual_t)max;
}

template <typename T>
void partial_reduce_residual_avg(Bitmap<T> &bmp,
				 vector<vector<noisy<T> > > &row,
				 vector<vector<T> > &diff,
				 vector<residual_t> &res,
				 long cluster_index
				 )
{
  long tmp, tmpd, sum = 0;
  for (unsigned int i=0; i<row[cluster_index].size();i++) {
    tmp = _ABS((long)bmp.get(row[cluster_index][i].c,row[cluster_index][i].r) - (long)row[cluster_index][i].original_pixel);
    tmpd = _ABS(tmp - (long)diff[cluster_index][i]);
    sum += tmpd;
    /*
    cerr << "(" << row[cluster_index][i].c << "," << row[cluster_index][i].r << ") "
	 << "orig: " << (long)row[cluster_index][i].original_pixel
	 << ", new: "<< (long)bmp.get(row[cluster_index][i].c,row[cluster_index][i].r)
	 << ", D(p,i) = " << tmp << ", D(p, i-1) = " << (long)diff[cluster_index][i] << ", R(p,i) = " << tmpd
	 << " [sumR(set #" << cluster_index << ", i): " << sum << "]" << endl;
    */
    diff[cluster_index][i] = (T)tmp;
  }
  res[cluster_index]= (residual_t)sum / (residual_t)row[cluster_index].size();
}



/*
  ------------------
  residual reduction
  ------------------
*/
template <typename T>
residual_t residual_avg(vector<residual_t> &sets_residuals, vector<vector<T> > &diff, unsigned int n_noisy) {
  residual_t sum = 0;
  for (unsigned int k=0; k<sets_residuals.size(); ++k)
    sum += (residual_t)diff[k].size() * sets_residuals[k];
  return (sum / n_noisy);
}

template <typename T>
residual_t residual_max(vector<residual_t> &sets_residuals/*, unsigned int n_noisy*/) {
  residual_t max = 0;
  for (unsigned int k=0; k<sets_residuals.size(); ++k)
    max = _MAX(max, sets_residuals[k]);
  return max;
}



/*
  -------------------------------
  single-block residual reduction
  -------------------------------
*/
//max
template <typename T>
residual_t reduce_residual(Bitmap<T> &bmp, vector<noisy<T> > &ns, vector<T> &diff) {
  long tmp, tmpd, max=0;
  long iter=0;
  for (unsigned int j=0; j<ns.size();j++) {
    tmp = _ABS((long)bmp.get(ns[j].c,ns[j].r) - (long)(ns[j].original_pixel));
    tmpd = _ABS((tmp - (long)diff[iter]));
    max = _MAX(max,tmpd);
    diff[iter++] = (T)tmp;
  }
  cerr << "Residual " << (residual_t) max << "\n";
  return (residual_t)max;
}
#endif
