#ifndef __VIDEOTASKTYPES_HPP__
#define __VIDEOTASKTYPES_HPP__
#include <opencv/cv.h>
#include "bitmap.hpp"
#include "control_structures.hpp"

template <typename T>
class Bitmap_cv : public Bitmap<T> {
public:
  //create 8-bit grayscale Bitmap from opencv IplImage
  Bitmap_cv(IplImage *img) : Bitmap<T>(img->width,img->height) {
    if ((img) && (img->depth==IPL_DEPTH_8U))
      for (int i = 0; i < img->height; ++i) {
	//uchar *tmp = (uchar *) (img->imageData + i*img->widthStep);
	for (int j = 0; j < img->width; ++j)
	  //Bitmap<T>::set(j,i,(T) tmp[j]);
	  Bitmap<T>::set(j,i,(T)(((uchar *)(img->imageData + i*img->widthStep))[j]));
      }
    else
      cerr << "Bitmap_cv: conversion not supported\n";
  }

  //create grayscale opencv IplImage from 8-bit grayscale Bitmap
  IplImage * createImg_cv(int depth,int nChannels) {
    if ((depth!=IPL_DEPTH_8U) || (depth!=1)) {
      IplImage *myimg = NULL;
      myimg = cvCreateImage(cvSize(this->width(), this->height()), depth,nChannels);   
      for (int i = 0; i < myimg->height; ++i)
	for (int j = 0; j < myimg->width; ++j)
      ((uchar *)(myimg->imageData + i*myimg->widthStep))[j] = (uchar)(this->get(j,i));
      return myimg;
    }
    else
      cerr << "createImg_cv: conversion not supported\n";
    return NULL;
  }
};

#endif

