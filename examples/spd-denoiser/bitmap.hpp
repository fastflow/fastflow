/*
  Implementation of a Bitmap with T-typed pixels.
  Pixels are read from a bmp-file and casted to the internal T-type.
*/

#ifndef _SPD_BITMAP_HPP_
#define _SPD_BITMAP_HPP_

//#include <limits.h>
#include "definitions.h"
#include <iostream>
#include <stdlib.h>
//#include <vector>
using namespace std;

#define BMPOFFSET (sizeof(T)*2+sizeof(short)+sizeof(bool)*2)
#define BMPALIGN 8

template <typename T = grayscale>
struct bmp_point {
  T value;
#if defined BORDER || defined FLAT
  T backup;
  short block_index; // to be moved in ctrl? 
  bool has_backup; // to be moved in ctrl?
  char padding[BMPALIGN - (BMPOFFSET % BMPALIGN)];
#endif
  bool noisy;
 
  bmp_point() {
    noisy = 0;
#if defined BORDER || defined FLAT
    has_backup = false;
#endif
  }
};

template <typename T = grayscale>
class Bitmap {
public:
  Bitmap() {}

  Bitmap(bmp_size_t w, bmp_size_t h) :
    columns(w), rows(h)
  {
    allocate();
  }

  ~Bitmap() {
    for(bmp_size_t r=0; r<rows; r++)
      free(bitmap[r]);
    free(bitmap);
  }

#if defined BORDER 
  inline void set_block_index(bmp_size_t x, bmp_size_t y, short index) {
    bitmap[y][x].block_index = index;
  }
  inline short get_block_index(bmp_size_t x, bmp_size_t y) {
    return(bitmap[y][x].block_index);
  }
  inline T get_with_backup(bmp_size_t x, bmp_size_t y) {
	return (bitmap[y][x].has_backup ? bitmap[y][x].backup:bitmap[y][x].value);
  }
#elif defined FLAT
  inline T get_with_backup(bmp_size_t x, bmp_size_t y) {
	return (bitmap[y][x].has_backup ? bitmap[y][x].backup:bitmap[y][x].value);
  }
#else
  inline T get_with_backup(bmp_size_t x, bmp_size_t y) {
	return (bitmap[y][x].value);
  }
#endif

  inline T get(bmp_size_t x, bmp_size_t y) {
    return bitmap[y][x].value;
  }

  inline void set(bmp_size_t x, bmp_size_t y, T v) {
    bitmap[y][x].value = v;
  }

  inline void set_noisy(int x, int y) {
    bitmap[y][x].noisy = 1;
  }

  inline int get_noisy(int x, int y) {
    return bitmap[y][x].noisy;
  }

  inline bmp_size_t height() {
    return rows;
  }

  inline bmp_size_t width() {
    return columns;
  }

#if defined FLAT || defined BORDER
  inline void backup(bmp_size_t x, bmp_size_t y) {
    bitmap[y][x].backup = bitmap[y][x].value;
    bitmap[y][x].has_backup = true;
  }

  inline bool has_backup(bmp_size_t x, bmp_size_t y) {
    return (bitmap[y][x].has_backup);
  }
#endif

  /*
  unsigned int get_resolution() {
    return bytes_per_pixel * 8;
  }
  */

  //read a bmp-file, without considering the color palette
  void read_8bit_nopalette(FILE *fp){
    read_init(fp);
    //bmp_size_t bytes_per_row = bytes_per_pixel * columns;
    bmp_size_t t = columns + (columns % 4); //padding
    int k = 1078; //first pixeldata byte of the row
    //pixel order: up -> down, left -> right
    unsigned char * buffer = (unsigned char *) malloc(columns * sizeof(unsigned char));
    for(bmp_size_t r = 0; r < rows; r++) {
      fseek(fp, k * sizeof(unsigned char), SEEK_SET);
      fread(buffer, sizeof(unsigned char), columns, fp);
      for(bmp_size_t c=0; c<columns; c++)
	set(c, rows-r-1, T(buffer[c])); //cast to internal pixel-type
      k += t; //next row
    }
    free(buffer);
  }
  
protected:
  struct bmp_point<T> **bitmap;
  bmp_size_t columns, rows;

private:
  //unsigned int bytes_per_pixel;

  void allocate() {
    struct bmp_point<T> ***p = &bitmap;
    posix_memalign((void **)p, ALIGNMENT, rows * sizeof(struct bmp_point<T> *));
    for(bmp_size_t r=0; r<rows; r++) {
      struct bmp_point<T> **p_ = bitmap + r;
      posix_memalign((void **)p_, ALIGNMENT, columns * sizeof(struct bmp_point<T>));
      for(bmp_size_t c=0; c<columns; c++) {
	bmp_point<T> *pp = *p_ + c;
	pp = new (pp) struct bmp_point<T>();
      }
    }
  }

  //read picture sizes and bytes-per-pixel, then allocate
  void read_init(FILE *fp) {
    //sizes: bytes [19-26]
    fseek(fp, 18 * sizeof(unsigned char), SEEK_SET);
    fread(&columns, 4 * sizeof(unsigned char), 1, fp);
    fseek(fp, 22 * sizeof(unsigned char), SEEK_SET);
    fread(&rows, 4 * sizeof(unsigned char), 1, fp);

    //bit-per-pixel: bytes [28-29]
    /*
    fseek(fp, 28 * sizeof(unsigned char), SEEK_SET);
    fread(&bytes_per_pixel, 4 * sizeof(unsigned char), 1, fp);
    bytes_per_pixel /= 8;
    if(bytes_per_pixel > sizeof(T))
      cerr << "warning: resolution (" << bytes_per_pixel * 8 << " bit) too high" << endl;
    */

    allocate();
  }
};


#endif
