#ifndef _SPD_NOISE_DETECTION_HPP_
#define _SPD_NOISE_DETECTION_HPP_
//#include "noise_detection.h"
#include "definitions.h"
#include "bitmap.hpp"
#include "parameters.h"
#include <algorithm>
using namespace std;

//control structures
//typedef grayscale_ctrl *ctrl_row;

template <typename T>
int ctrl_px(Bitmap<T> &bmp, grayscale_ctrl a[], unsigned int w, bmp_size_t y, bmp_size_t x, bmp_size_t width, bmp_size_t height) {
  int cw(w/2);
  y -= cw;
  x -= cw;
  unsigned int n = 0;
  for(bmp_size_t r=0; r<w; r++) //rows
    for(bmp_size_t c=0; c<w; c++) //columns
      if((x+c >= 0) && (x+c < width) && (y+r >= 0) && (y+r < height))
	a[n++] = (long)bmp.get(x+c, y+r);
  return n;
}

//AMF filter on grayscale pixel
inline bool is_noisy(
	      grayscale center,
	      bmp_size_t r,
	      bmp_size_t c,
	      unsigned int w_max,
	      long *c_array,
	      Bitmap<grayscale> &bmp,
	      bmp_size_t width,
	      bmp_size_t height
	      )
{
  if (center==SALT || center==PEPPER) {
    for(unsigned int ws=7, wi=0; ws<w_max; ws+=2, ++wi) {
      int ac = ctrl_px<grayscale>(bmp, c_array, ws, r, c, width, height);
      sort(c_array, c_array + ac);
      long min_cp = c_array[0];
      long med_cp = c_array[ac / 2];
      long max_cp = c_array[ac - 1];
      if((!(min_cp==med_cp && med_cp==max_cp)) || (ws>w_max))
		//not homogeneous zone
		if ((center==min_cp || center==max_cp || ws>w_max))
		  return true; //noisy
    }
  }
  return false; //not noisy
}

template <typename T>
bmp_size_t find_noisy_partial(
			      Bitmap<T> &bmp,
#ifdef CLUSTER
			      Bitmap_ctrl<T> &bmp_ctrl,
#endif
			      unsigned int w_max,
			      vector<noisy<T> > &noisy_pixels,
			      bmp_size_t first_row,
			      bmp_size_t last_row,
			      bmp_size_t width
			      )
{
  bmp_size_t n_noisy = 0;
  //linearized control window
  long c_array[MAX_WINDOW_SIZE * MAX_WINDOW_SIZE];
  bmp_size_t height = bmp.height();

  for(bmp_size_t r=first_row; r<=last_row; ++r) {
    for(bmp_size_t c=0; c<width; c++) {
      T center = bmp.get(c, r);
      if(is_noisy(center, r, c, w_max, c_array, bmp, width, height)) {
	noisy_pixels.push_back(noisy<T>(c, r, center));
	bmp.set_noisy(c, r);
#ifdef CLUSTER
	bmp_ctrl.set_noisy_i(c, r, n_noisy);
	// annote dependencies
	if (c > 0 && bmp.get_noisy(c-1, r)) {
	  bmp_size_t left_noisy_i = bmp_ctrl.get_noisy_i(c-1, r);
	  noisy_pixels[left_noisy_i].right_noisy = true;
	  noisy_pixels[left_noisy_i].right_noisy_i = n_noisy;
	}
	if (r > 0 && bmp.get_noisy(c, r-1)) {
	  bmp_size_t down_noisy_i = bmp_ctrl.get_noisy_i(c, r-1);
	  noisy_pixels[down_noisy_i].up_noisy = true;
	  noisy_pixels[down_noisy_i].up_noisy_i = n_noisy;
	}
#endif
	++n_noisy;
      }
    }
  }
  //cerr << "N. noisy " << n_noisy << "\n";
  return n_noisy;
}
#endif
