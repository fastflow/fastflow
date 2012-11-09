#ifndef _SPD_FF_ACCEL_HPP_
#define _SPD_FF_ACCEL_HPP_
#include "pow_table.hpp"
#include "fuy.hpp"
#include "convergence.hpp"
#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <algorithm>
#include <vector>
#include <cmath>
using namespace std;

/*
  -------------------------------------------------------
  Emitter (and convergence-checker) for CONTINUOUS scheme
  -------------------------------------------------------
*/
#ifdef CONTINUOUS
template <typename T>
class FFE_EmitConv: public ff::ff_node {
public:
  FFE_EmitConv(const unsigned long long streamlen, const float epsilon_residual,
	       const unsigned long long n_noisy, vector<vector<T> > &diff,
	       vector<residual_t> &sets_residuals, unsigned int &cycles, unsigned int max_cycles):
    streamlen(streamlen),epsilon_residual(epsilon_residual),n_noisy(n_noisy),
    count(0),cycles(cycles),max_cycles(max_cycles),diff(diff),sets_residuals(sets_residuals),old_residual(0.0),sum(0.0) { };

  // diff is only used for its size, which is constant. 
  // the code can be optimised. 

  int svc_init() {
    for(unsigned long long j=1; j<=streamlen; j++)
      ff_send_out((void *) j);
    return 0;
  }

  void * svc(void * task) {
    long t = (long) task;

    if (!task) 
      return GO_ON;

    count++;
    if (count >= streamlen) {
      current_residual = sum / n_noisy;
      sum = 0.0;
      delta_residual = _ABS((old_residual - current_residual));
      //cout << "current residual " << sum << " delta " << delta_residual << " old_residual " << old_residual << "\n";
      old_residual = current_residual;
      ++cycles;
      count=0;
	
    }
    else
      sum += diff[t-1].size() * sets_residuals[t-1];

    //convergence?
    bool fix = (count == 0);
#ifndef CONV_FIXED
    fix = fix && ((delta_residual < epsilon_residual) || cycles == max_cycles);
#else
    fix = fix && (cycles == max_cycles);
#endif
    if(fix)
      //cerr << "N of cycles " << cycles << endl;
      return NULL;
    else
      return task;
  }

private:
  const unsigned long long streamlen;
  const float epsilon_residual;
  const bmp_size_t n_noisy;
  unsigned long long count;
  unsigned int &cycles, max_cycles;
  vector<vector<T> > &diff;
  vector<residual_t> &sets_residuals;
  float old_residual, current_residual,delta_residual;
  float sum;
};
#endif // CONTINOUS





/*
  -----------------------------------------
  DENOISER-worker for non-CONTINUOUS scheme
  -----------------------------------------
*/
template <typename T>
class FFW_Filter: public ff::ff_node {
public:
  FFW_Filter(
	 Bitmap<T> &bmp,
	 float alfa,
	 float beta,
	 bmp_size_t width,
	 bmp_size_t height,
	 vector<vector<noisy<T> > > &clusters,
	 vector<vector<T> > &diff,
	 vector<residual_t> &res
	 )
    :
    bmp(bmp),
    alfa(alfa), beta(beta),
    width(width), height(height),
    clusters(clusters),
    diff(diff), res(res)
  {
    pow_table_alfa = new pow_table(alfa);
  }

  ~FFW_Filter() {
    delete pow_table_alfa;
  }

  void * svc(void * _task) {
    long task = (long)_task;

    if((unsigned long)task <= clusters.size()) {
      --task;
    //pass
    fuy_set(task);

    //compute the residual for the noisy set
#ifdef AVG_TERMINATION
    partial_reduce_residual_avg<T>(bmp, clusters, diff, res, task);
#else
    partial_reduce_residual_max<T>(bmp, clusters, diff, res, task);
#endif
    }

    else {
#ifdef FLAT
      //backup task
      --task;
      task -= clusters.size();
      for(unsigned int j=0; j<clusters[task].size(); ++j)
	bmp.backup(clusters[task][j].c, clusters[task][j].r);
#else
      //error
#endif
    }

    return _task;
  }

private:
  Bitmap<T> &bmp;
  float alfa;
  float beta;
  int width;
  int height;
  vector<vector<noisy<T> > > &clusters;
  vector<vector<T> > &diff;
  vector<residual_t> &res;
  //PowFast::PowFast const &mypow;
  pow_table *pow_table_alfa;

  void fuy_set(long set_i) {
    vector<noisy<T> > &set = clusters[set_i];
    for(unsigned int i=0; i<set.size(); i++)
      fuy(bmp, set[i], /*set_i, */width, height, /* alfa, */ beta, /* i, */ 
	  *pow_table_alfa);
  }
};





/*
  ---------------
  DETECTOR-worker
  ---------------
 */
typedef struct detection_task {
  bmp_size_t first_row;
  bmp_size_t last_row;
  long out_i;

  detection_task(bmp_size_t fr, bmp_size_t lr, long o_i) {
    first_row = fr;
    last_row = lr;
    out_i = o_i;
  }
} detection_task_t;

template <typename T>
class Worker_detector: public ff::ff_node {
public:
  Worker_detector(
		  Bitmap<T> &bmp,
		  Bitmap_ctrl<T> &bmp_ctrl,
		  bmp_size_t width,
		  vector<vector<noisy<T> > >&outs,
		  unsigned int w_max
		  )
    :
    bmp(bmp), bmp_ctrl(bmp_ctrl),
    width(width),
    outs(outs),
    w_max(w_max)
  {}

  void * svc(void * task) {
    detection_task_t *t = (detection_task_t *)task;
    bmp_size_t first_row = t->first_row;
    bmp_size_t last_row = t->last_row;
    long out_i = t->out_i;
    delete t;
    outs[out_i].reserve((last_row - first_row + 1) * width);
    find_noisy_partial<T>(bmp, w_max, outs[out_i], first_row, last_row, width);
    return GO_ON;
  }

private:
  Bitmap<T> &bmp;
  Bitmap_ctrl<T> &bmp_ctrl;
  bmp_size_t width;
  vector<vector<noisy<T> > >&outs;
  unsigned int w_max;
};
#endif
