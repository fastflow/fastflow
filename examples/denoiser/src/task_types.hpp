#ifndef _TASK_TYPES_HPP_
#define _TASK_TYPES_HPP_

//denoise task
typedef struct denoise_task {
  ~denoise_task() { //TODO: check delete, delete[], free
    if(input) delete input;
    if(output) delete output;
    if(noisymap) delete noisymap;
    if(noisy) delete noisy;
  }

  denoise_task() {
	  input = output = NULL;
	  noisymap = NULL;
	  noisy = NULL;
	  n_noisy = n_cycles = width = height = 0;
  }

  unsigned char *input, *output;
  int *noisymap; //-1 if not noisy, otherwise the original pixel
  unsigned int *noisy;
  unsigned int n_noisy;
  unsigned int n_cycles;
  unsigned int width, height;
  void *kernel_params;
  bool fixed_cycles;
} denoise_task_t;

#endif
