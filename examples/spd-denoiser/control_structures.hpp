#ifndef _SPD_CONTROL_STRUCTURES_HPP_
#define _SPD_CONTROL_STRUCTURES_HPP_

#include "definitions.h"
#include "bitmap.hpp"
#include <vector>
using namespace std;

/*
  ------------------
  BMP-control bitmap
  ------------------
*/
/*
  typedef struct point_info {
  bmp_size_t noisy_i;
  } point_info_t;
*/
template <typename T = grayscale>
class Bitmap_ctrl : public Bitmap<bmp_size_t> {
public:
  Bitmap_ctrl(Bitmap<T> &copy) : Bitmap<bmp_size_t>(copy.width(), copy.height()) {}

  bmp_size_t get_noisy_i(bmp_size_t c, bmp_size_t r) {
    return bitmap[r][c].value;
  }

  void set_noisy_i(bmp_size_t c, bmp_size_t r, bmp_size_t noisy_i) {
    bitmap[r][c].value = noisy_i;
  }
};





/*
  -----------
  noisy pixel
  -----------
*/
template <typename T>
struct noisy {
  bmp_size_t r;
  bmp_size_t c; 
  T original_pixel;
#ifdef CLUSTER
  //unsigned char nextvalue; // MA cache opt
  long cluster;
  //noisy dependencies
  bool right_noisy, up_noisy;
  bmp_size_t right_noisy_i, up_noisy_i;
#endif

  noisy(bmp_size_t col, bmp_size_t row, grayscale orig)
    : r(row), c(col), original_pixel(orig)
  {
#ifdef CLUSTER
    cluster = -1;
    right_noisy = up_noisy = false;
    right_noisy_i = up_noisy_i = 0; // was -1 
#endif
  }

  noisy() {} //dummy constructor


  bool operator < (const noisy& p1) const {
    return (r < p1.r || (c < p1.c && r == p1.r));
  }
};
#endif
