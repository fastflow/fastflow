#ifndef _SPD_PIPE_FF_
#define _SPD_PIPE_FF_

//FF
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <ff/node.hpp>
//#include <ff/ocl/mem_man.h>
using namespace ff;

// #include <fstream>
// using namespace std;

#include "utils.hpp"
#include "noise_detection.hpp"
#include "kernel_ff/fuy.hpp"
#include "kernel_ff/pow_table.hpp"

#define EPS_STOP 1e-04f



/*
  DETECT STAGE (seq)
 */
class Detect_FF: public ff_node {
public:
  Detect_FF(unsigned int w_max): w_max(w_max) {}

  void * svc(void * task) {
#ifdef TIME
    unsigned long time_ = get_usec_from(0);
#endif
    task_frame *t = (task_frame *)task;
    vector<unsigned int> noisy_;
    t->noisymap = new int[t->height * t->width];
    detect(noisy_, t->noisymap, t->in, t->height, t->width, w_max);
    t->n_noisy = noisy_.size();
    t->noisy = new unsigned int[t->n_noisy];
    for(unsigned int i=0; i<t->n_noisy; ++i)
      (t->noisy)[i] = noisy_[i];
#ifdef TIME
    t->svc_time_detect = get_usec_from(time_);
#endif
    return task;
  }

private:
  unsigned int w_max;

  void detect(
	      vector<unsigned int> &noisy,
	      int *noisymap,
	      unsigned char *im,
	      unsigned int height,
	      unsigned int width,
	      unsigned int w_max
	      )
  {
    //linearized control window
    unsigned char c_array[MAX_WINDOW_SIZE * MAX_WINDOW_SIZE];
    unsigned int first_row = 0, last_row = height-1;
    for(unsigned int r=first_row; r<=last_row; ++r) {
      for(unsigned int c=0; c<width; c++) {
	unsigned int idx = r * width + c;
	unsigned char center = im[idx];
	if(is_noisy<unsigned char>(center, r, c, w_max, c_array, im, width, height)) {
	  noisy.push_back(idx);
	  noisymap[idx] = center;
	}
	else
	  noisymap[idx] = -1;
      }
    }
  }
};





class Denoise_FF_Bypass: public ff_node {
public:
  Denoise_FF_Bypass() {}
   
  void * svc(void * task) {
    task_frame *t = (task_frame *)task;
    unsigned char *input = t->in;
    unsigned char *out = t->out;
    unsigned int *noisy = t->noisy;
    int *noisymap = t->noisymap;
    unsigned int n_noisy = t->n_noisy;
    unsigned int height = t->height;
    unsigned int width = t->width;
    t->out = new unsigned char[height * width];
    for(unsigned int i=0; i<height; ++i)
      for(unsigned int j=0; j<width; ++j)
	out[i*width + j] = input[i*width + j];

    return task;
  }
};



typedef struct task_denoise_acc {
  task_frame *task;
  unsigned int idx, first, last;
  unsigned char *diff;
  float *block_residuals;

  task_denoise_acc(unsigned int idx, task_frame *task, unsigned int first, unsigned int last, unsigned char *diff, float *block_residuals) :
    idx(idx), task(task), first(first), last(last), diff(diff), block_residuals(block_residuals) {}
} task_denoise_acc;

class DenoiseW : public ff_node {
public:
  DenoiseW(float alfa, float beta) : alfa(alfa), beta(beta)
  {
    pt = new pow_table(alfa);
  }

  ~DenoiseW() {
    delete pt;
  }

  void *svc(void *task) {
    task_denoise_acc *t = (task_denoise_acc *)task;
    task_frame *tf = t->task;
    //fuy the block
    float residual = 0;
    for(unsigned int i = t->first; i != t->last+1; ++i) {
      unsigned int im_idx = tf->noisy[i];
      fuy(tf->out, tf->in, im_idx, tf->noisymap, tf->width, tf->height, alfa, beta, *pt);
      //residuals
      unsigned char newdiff = (unsigned char)(_ABS((int)(tf->out[im_idx]) - tf->noisymap[im_idx]));
      residual += (float)(_ABS((int)newdiff - (int)(t->diff[i])));
      t->diff[i] = newdiff;
    }
    t->block_residuals[t->idx] = residual;
    return task;
  }

private:
  float alfa, beta;
  pow_table *pt;
};



#define BLOCK_SIZE 128
class Denoise_FF : public ff_node {
public:
  Denoise_FF(float alfa, float beta, unsigned int n_workers) : n_workers(n_workers) {
    //setup farm-acc.
    denoise_farm_acc = new ff_farm<>(true);
    for(unsigned int i=0; i<n_workers; ++i)
      denoise_workers.push_back(new DenoiseW(alfa, beta));
    denoise_farm_acc->add_workers(denoise_workers);
  }

  void *svc(void *task) {
#ifdef TIME
    unsigned long time_ = get_usec_from(0);
#endif
    task_frame *t = (task_frame *)task;
    //copy input into output
    t->out = new unsigned char[t->height * t->width];
    memcpy(t->out, t->in, sizeof(unsigned char)*t->height*t->width);

    if(t->n_noisy > 0) {
      //compute blocks
      unsigned int n_blocks = (t->n_noisy + BLOCK_SIZE - 1) / BLOCK_SIZE;

      //allocate memory for residuals
      unsigned char *diff = new unsigned char[t->n_noisy];
      memset(diff, 0, sizeof(unsigned char)*(t->n_noisy));
      float *block_residuals = new float[n_blocks];
      for(unsigned int i=0; i<n_blocks; ++i)
	block_residuals[i] = 0;
      vector<task_denoise_acc *> tasks(n_blocks, NULL);

      //prepare the acc-tasks
      unsigned int first = 0;
      unsigned int i = 0;
      for(i=0; i<n_blocks-1; ++i) {
	tasks[i] = new task_denoise_acc(i, t, first, first + BLOCK_SIZE - 1, diff, block_residuals);
	first += BLOCK_SIZE;
      }
      tasks[i] = new task_denoise_acc(i, t, first, t->n_noisy-1, diff, block_residuals);

      //start loop
      float residual=0, old_residual, delta;
      bool fix = false;
      while(!fix) {
	++t->cycles;
	denoise_farm_acc->run_then_freeze();
	//offload blocks
	for(i=0; i<n_blocks; ++i)
	  denoise_farm_acc->offload((void *)tasks[i]);
	//join
	denoise_farm_acc->offload((void *)ff::FF_EOS);
	denoise_farm_acc->wait_freezing();
	//reduce residuals
	old_residual = residual;
	residual = 0;
	for(i=0; i<n_blocks; ++i) {
	  residual += block_residuals[i];
	}
	residual /= t->n_noisy;
	
	//check convergence
	delta = _ABS(residual - old_residual);
	//cerr << "delta = " << delta << endl;
	fix = t->cycles == t->max_cycles;
	if(!(t->fixed_cycles))
	  fix |= delta < EPS_STOP;

	if(!fix) {
	  //copy output into input
	  memcpy(t->in, t->out, sizeof(unsigned char)*t->height*t->width);
	}
      } //end loop

      for(i=0; i<n_blocks; ++i)
	delete tasks[i];
      delete[] diff;
      delete[] block_residuals;
    }

#ifdef TIME
    t->svc_time_denoise = get_usec_from(time_);
#endif
    return task;
  }

  ~Denoise_FF() {
    for(unsigned int i=0; i<n_workers; ++i)
      delete (DenoiseW *)(denoise_workers[i]);
    delete denoise_farm_acc;
  }

private:
  ff_farm<> *denoise_farm_acc;
  vector<ff_node *> denoise_workers;
  unsigned int n_workers;
};





typedef struct pipe_components {
  Detect_FF *detect_stage;
  Denoise_FF *denoise_stage;
} pipe_components_t;

void setup_pipe(ff_pipeline &pipe, pipe_components_t &comp, unsigned int w_max, float alfa, float beta, unsigned int CORE_COUNT) {
  comp.detect_stage = new Detect_FF(w_max);
  comp.denoise_stage = new Denoise_FF(alfa, beta, CORE_COUNT);
  //comp.denoise_stage = new Denoise_FF_Bypass();
  //add stages
  pipe.add_stage(comp.detect_stage);
  pipe.add_stage(comp.denoise_stage);
}

void clean_pipe(pipe_components_t &comp) {
  delete comp.detect_stage;
  delete comp.denoise_stage;
}
#endif
