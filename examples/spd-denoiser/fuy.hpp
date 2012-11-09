#ifndef __FUY_HPP__
#define __FUY_HPP__

#include "definitions.h"
#include "bitmap.hpp"
#include "control_structures.hpp"
#include "pow_table.hpp"
#include "utils.hpp"
//#include <cmath>



template <typename T>
struct point_ctrl {
  T value;
  unsigned char noisy;
};



//fill the 4-closest window for a pixel
template <typename T>
inline unsigned int closest(
			    Bitmap<T> &bmp,
			    point_ctrl<T> *fc,
			    bmp_size_t y,
			    bmp_size_t x,
			    bmp_size_t width,
			    bmp_size_t height
			    //long set_i
			    )
{
  unsigned int nc = 0;
  //sotto
  if(y > 0) {
    fc[nc].value = bmp.get_with_backup(x, y-1);
    fc[nc].noisy = bmp.get_noisy(x, y-1);
    ++nc;
  }
  //sx
  if(x > 0) {
    fc[nc].value = bmp.get_with_backup(x-1, y);
    fc[nc].noisy = bmp.get_noisy(x-1, y);
    ++nc;
  }
  //dx
  if(x < width-1) {
    fc[nc].value = bmp.get_with_backup(x+1, y);
    fc[nc].noisy = bmp.get_noisy(x+1, y);
    ++nc;
  }
  //sopra
  if(y < height-1) {
    fc[nc].value = bmp.get_with_backup(x, y+1);
    fc[nc].noisy = bmp.get_noisy(x, y+1);
    ++nc;
  }

  return nc;
}



//filter over a 8-bit grayscale pixel
inline void fuy(
		Bitmap<grayscale> &bmp,
		noisy<grayscale> &noisy,
		/*long set_i,*/
		bmp_size_t width,
		bmp_size_t height,
		//float alfa, // useless - remove
		float beta,
		//bmp_size_t i, // useless - remove
		pow_table &pt
		)
{
  //get the pixel and the four closest
  bmp_size_t noisy_y = noisy.r;
  bmp_size_t noisy_x = noisy.c;
  grayscale pixel = bmp.get(noisy_x, noisy_y);
  point_ctrl<grayscale> fc[4]; //up-to-4 closest
  unsigned int nc = closest<grayscale>(bmp, fc, noisy_y, noisy_x, width, height/*, set_i*/);

  //compute the correction
  grayscale u = 0;
  float S;
  float Fu, Fu_min = 0.0f, Fu_prec = 256.0f;
  float beta_ = beta / 2;
  for(int uu=0; uu<256; ++uu) {
    u = (grayscale) uu;
    Fu = 0.0f;
    S = 0.0f;
	
    for(unsigned int h=0; h<nc; ++h)
      S += (2-fc[h].noisy) * pt((grayscale)_ABS(((long)u - (long)fc[h].value)));
    Fu += ((grayscale)_ABS((long)u - (long)pixel) + (beta_) * S);
    if(Fu < Fu_prec)
      Fu_min = u;
    Fu_prec = Fu;
  }

  //actual correction
  bmp.set(noisy_x, noisy_y, (grayscale)Fu_min);
}
#endif 
