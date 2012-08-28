#ifndef _FF_ACCEL_VIDEO_HPP_
#define _FF_ACCEL_VIDEO_HPP_
#include "bitmap.hpp"
#include "control_structures.hpp"
#include <utils.h>
#include <pow_table.hpp>
#include <convergence.hpp>
#include <noise_detection.hpp>

#include <map>
#include <queue>

#include <ff/node.hpp>
using namespace ff;

//pipeline task
template <typename T>
struct noisy_img_task {
  Bitmap<T> *bmp;

  //detection
  int set_index;
  vector<noisy<T> > *the_noisy_set;
  bmp_size_t first_row;
  bmp_size_t last_row;

  //input task
  noisy_img_task(Bitmap<T> *bmp): bmp(bmp) {}

  //detection task
  noisy_img_task(Bitmap<T> *bmp, int set_index, bmp_size_t first_row, bmp_size_t last_row)
    : bmp(bmp), set_index(set_index), first_row(first_row), last_row(last_row) {}

  //denoising task
  noisy_img_task(Bitmap<T> *bmp, vector<noisy<T> > *the_noisy_set)
    : bmp(bmp), the_noisy_set(the_noisy_set) {}
};



/*
  --------------------------
  sequential detection stage
  --------------------------
*/
template <typename T>
class Detect: public ff_node {
public:

  Detect(unsigned int w_max): w_max(w_max) {}

  void * svc(void * task) {
    noisy_img_task<T> *t = (noisy_img_task<T> *)task;
    //cerr << "Detect stage img width " << t->bmp->width() <<  " height " << t->bmp->height() << "\n";
#ifdef TIME
    long t_det = get_usec_from(0);
#endif
    t->the_noisy_set = new vector<noisy<T> >;
    t->the_noisy_set->reserve((t->bmp->height())*(t->bmp->width())/10);
    find_noisy_partial<T>(*(t->bmp), w_max, *(t->the_noisy_set), 0, t->bmp->height()-1, t->bmp->width());
#ifdef TIME
    t_det = get_usec_from(t_det)/1000;
    cerr << "Detect Time :" << t_det << " (ms) " << "Noisy pixels: " << t->the_noisy_set->size() << endl;
#endif
    return task;
  }

private:
  unsigned int w_max;
};



/*
  ----------------------
  detection-farm emitter
  ----------------------
*/
template <typename T>
class FFE_detect: public ff_node {
public:
  FFE_detect(unsigned int n_sets): n_sets(n_sets) {}

  void * svc(void * task) {
    noisy_img_task<T> *t = (noisy_img_task<T> *)task;
    Bitmap<T> *bmp = t->bmp;
    bmp_size_t height = bmp->height();
    delete t;

    //split and send
    bmp_size_t step_rows = (height + n_sets - 1) / n_sets; //round up
    bmp_size_t first_row = 0, last_row;
    unsigned int set_index = 0;
    while(true) {
      last_row = min(first_row + step_rows, height - 1);
      ff_send_out(new noisy_img_task<T>(bmp, set_index++, first_row, last_row));
      if(last_row == (height - 1))
	break;
      first_row = last_row + 1;
    }

    return GO_ON;
  }

private:
  unsigned int n_sets;
};



/*
  ---------------------
  detection-farm worker
  ---------------------
*/
template <typename T>
class FFW_detect: public ff_node {
public:
  FFW_detect(unsigned int w_max): w_max(w_max) {}

  void * svc(void * task) {
    noisy_img_task<T> *t = (noisy_img_task<T> *)task;

    t->the_noisy_set = new vector<noisy<T> >;
    t->the_noisy_set->reserve((t->bmp->height())*(t->bmp->width())/10);
    find_noisy_partial<T>(*(t->bmp), w_max, *(t->the_noisy_set), t->first_row, t->last_row, t->bmp->width());
    //cerr << "[worker #" << this << "] sending set " << t->set_index << " for bmp {" << t->bmp << "}" << endl;

    return task;
  }

private:
  unsigned int w_max;
};



/*
  ------------------------
  detection-farm collector
  ------------------------
*/
template <typename T>
class FFC_detect: public ff_node {
public:
  FFC_detect(unsigned int n_sets): n_sets(n_sets) {}

  void * svc(void * task) {
    noisy_img_task<T> *t = (noisy_img_task<T> *)task;
    Bitmap<T> *bmp = t->bmp;

    //buffer
    if(!buffer.count(bmp)) {
      bmp_order.push(bmp);
      buffer[bmp] = vector<vector<noisy<T> > *>(n_sets, NULL);
      sets_counts[bmp] = 0;
    }
    buffer[bmp][t->set_index] = t->the_noisy_set;
    sets_counts[bmp] += 1;
    //cerr << "collected set " << t->set_index << " for bitmap {" << bmp << "}" << endl;
    delete t;

    //send completed denoising tasks
    while(true) {
      bmp = bmp_order.front();
      cerr << "visiting bmp {" << bmp << "}: " << sets_counts[bmp] << " sets" << endl;
      if(sets_counts[bmp] == n_sets) {
	cerr << "completed buffer for bmp {" << bmp << "}" << endl;
	//compute n_noisy
	unsigned int n_noisy = 0;
	for(unsigned int i=0; i<n_sets; ++i)
	  n_noisy += (buffer[bmp][i])->size();
	//merge
	vector<noisy<T> > &v = *(new vector<noisy<T> >(n_noisy));
	unsigned int vi = 0;
	for(unsigned int i=0; i<n_sets; ++i) {
	  for(unsigned int j=0; j<buffer[bmp][i]->size(); ++j)
	    v[vi++] = buffer[bmp][i]->at(j);
	  delete buffer[bmp][i];
	}
	buffer.erase(bmp);
	sets_counts.erase(bmp);
	bmp_order.pop();
	cerr << "will send task: " << bmp << ", " << &v << endl;
	ff_send_out(new noisy_img_task<T>(bmp, &v));
      }
      else
	break;
    }

    return GO_ON;
  }

private:
  unsigned int n_sets;
  queue<Bitmap<T> *> bmp_order;
  map<Bitmap<T> *, vector<vector<noisy<T> > *> > buffer;
  map<Bitmap<T> *, unsigned int> sets_counts;
};



/*
  --------------------------
  single-set denoising stage
  --------------------------
*/
template <typename T>
class Denoise: public ff_node {
public:
  Denoise(double alpha, double beta):alpha(alpha), beta(beta) {
    pow_table_alfa = new pow_table(alpha);
  }

  void * svc(void * task) {
    noisy_img_task<T> *t = (noisy_img_task<T> *)task;
    vector<noisy<T> > &set = *(t->the_noisy_set);
    vector<grayscale> diff(set.size(),0);
    int cur_residual, old_residual = 0;
#ifdef TIME
    int cycles = 0;
    long t_rec = get_usec_from(0);
#endif
    do {
      old_residual = cur_residual;
      for(unsigned int i=0; i<set.size(); ++i)
	fuy(*(t->bmp), set[i], /*0, */t->bmp->width(), t->bmp->height(), alpha, beta, i, *pow_table_alfa);
      cur_residual = reduce_residual<grayscale>(*(t->bmp), set, diff);
#ifdef TIME
      ++cycles;
#endif
    } while  (_ABS((old_residual - cur_residual)) > 0);
    delete t->the_noisy_set;
    t->the_noisy_set = NULL;
#ifdef TIME
    t_rec = get_usec_from(t_rec)/1000;
    cerr << "Denoising Time :" << t_rec << " (ms) Cycles " << cycles << "\n";
#endif
    
    return task;
  }

private:
  double alpha, beta;
  pow_table *pow_table_alfa;
};
#endif
