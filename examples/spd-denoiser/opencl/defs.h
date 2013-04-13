#ifndef _SPD_DEFS_H_
#define _SPD_DEFS_H_

#define SALT 255
#define PEPPER 0

#ifdef SPD_OCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#endif

#ifdef SPD_OCL
typedef cl_uchar pixel_t;
typedef cl_uint bmpsize_t;
typedef cl_int pixelext_t;
#endif
#ifdef SPD_FF
typedef unsigned char pixel_t;
typedef unsigned int bmpsize_t;
typedef int pixelext_t;
#endif

typedef struct task_frame {
  unsigned long frame_id;
  pixel_t *in, *out;
  bmpsize_t *noisy;
  bmpsize_t n_noisy, height, width;
  pixelext_t *noisymap;
  int cycles, max_cycles;
  bool fixed_cycles;
#ifdef TIME
  unsigned long svc_time_detect, svc_time_denoise;
#endif

  task_frame(unsigned long id, pixel_t *in, unsigned int height, unsigned int width, int max_cycles, bool fixed_cycles)
    : frame_id(id),
      in(in), out(NULL), noisy(NULL), noisymap(NULL),
      height(height), width(width), n_noisy(0),
      max_cycles(max_cycles), fixed_cycles(fixed_cycles), cycles(0) {}

  ~task_frame() {
    delete[] in;
    delete[] out;
    delete[] noisy;
    delete[] noisymap;
  }
} task_frame;
#endif
