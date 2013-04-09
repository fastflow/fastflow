#ifndef _SPD_CAST_HPP_
#define _SPD_CAST_HPP_

#if (defined SPD_OCL) && (defined __APPLE__) 
#include <OpenCL/opencl.h>
#endif

#if (defined SPD_OCL) && (not defined __APPLE__)
#include <CL/opencl.h>
#endif 

#include <opencv/cv.h>
using namespace cv;

#include <iostream>
using namespace std;

template <typename T>
T *Ipl2uchar(IplImage *in) {
  T *out = new T[in->height * in->width];
  for(int i=0; i<in->height; ++i)
    for(int j=0; j<in->width; ++j)
      out[i*(in->width) + j] = ((unsigned char *)(in->imageData + i*in->widthStep))[j];
  return out;
}

template <typename T>
IplImage *uchar2Ipl (T *in, unsigned int height, unsigned int width) {
  IplImage *myimg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
  for (unsigned int i = 0; i < height; ++i)
    for (unsigned int j = 0; j < width; ++j)
      ((unsigned char *)(myimg->imageData + i*myimg->widthStep))[j] = (unsigned char)(in[i * width + j]);
  return myimg;
}
#endif
