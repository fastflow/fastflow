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

#endif
