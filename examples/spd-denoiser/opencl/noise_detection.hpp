#ifndef _SPD_NOISE_DETECTION_HPP_
#define _SPD_NOISE_DETECTION_HPP_

#include <algorithm>
using namespace std;

#include "defs.h"
#include "parameters.h"

/**
 * @param w the control-window size
 * @param yy the control-window y-center
 * @param xx the control-window x-center
 * @return the number of valid pixels within the control-window
 */
template <typename T>
int ctrl_px(pixel_t *im, pixel_t *a, unsigned int w, unsigned int yy, unsigned int xx, unsigned int width, unsigned int height) {
  unsigned int cw(w/2);
  int n = 0;
  for(unsigned int r=0; r<w; r++) //rows
    for(unsigned int c=0; c<w; c++) //columns
      if(((xx + c) >= cw) && ((xx + c) < (width + cw)) && ((yy + r) >= cw) && ((yy + r) < (height + cw)))
	a[n++] = im[(yy + r - cw) * width + xx + c - cw];
  return n;
}

//AMF filter on grayscale pixel
template <typename T>
inline bool is_noisy(
		     T center,
		     unsigned int r,
		     unsigned int c,
		     unsigned int w_max,
		     T *c_array,
		     T *im,
		     unsigned int width,
		     unsigned int height
		     )
{
  if (center==SALT || center==PEPPER) {
    for(unsigned int ws=3; ws<w_max; ws+=2) {
      int ac = ctrl_px<T>(im, c_array, ws, r, c, width, height);
      sort(c_array, c_array + ac);
      T min_cp = c_array[0];
      T med_cp = c_array[ac / 2];
      T max_cp = c_array[ac - 1];
      if((!(min_cp==med_cp && med_cp==max_cp)) || (ws>w_max))
	//not homogeneous zone
	if (center==min_cp || center==max_cp || ws>w_max)
	  return true; //noisy
    }
  }
  return false; //not noisy
}
#endif
