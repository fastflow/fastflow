#ifndef _SPD_BMP_FINALIZER_HPP_
#define _SPD_BMP_FINALIZER_HPP_
#include "bitmap.hpp"
#include <iostream>
#include <vector>
using namespace std;

void write_8bit_nopalette(FILE *fp, Bitmap<grayscale> &bmp, bmp_size_t x, bmp_size_t y, string prefix){
  //open the output file
  //cout << "- WRITING" << endl;
  prefix.append(".bmp");
  FILE *fbmp = fopen(prefix.c_str(), "wb");
  if(fbmp == NULL){
    std::cerr << "Could not create the output file: " << prefix << endl;
    exit(1);
  }

  //copy headers
  unsigned char buf[1078];
  fseek(fp,0,SEEK_SET);
  fread(buf, sizeof(unsigned char), 1078, fp);
  fwrite(buf,sizeof(unsigned char), 1078, fbmp);
  fflush(fbmp);

  /*
  unsigned int bytes_per_pixel = bmp.get_resolution() / 8;
  unsigned int bytes_per_row = bytes_per_pixel * x;
  */

  //compute padding
  bmp_size_t p = x % 4;
  //int t = x + p;
  unsigned char * padding = (unsigned char *) malloc(sizeof(unsigned char) * p);
  for(bmp_size_t i=0; i<p; i++)
    padding[i] = (unsigned char)0;
  
  unsigned char * row_buf = (unsigned char *) malloc(sizeof(unsigned char) * x);
  for(bmp_size_t r=0; r<y; r++) {
    for(bmp_size_t c=0; c<x; c++)
      row_buf[c] = (unsigned char)bmp.get(c, y-r-1);
    fwrite(row_buf, sizeof(unsigned char), x, fbmp);
    fwrite(padding, sizeof(unsigned char), p, fbmp);
  }
  fflush(fbmp);
  fclose(fbmp);
  free(padding);
  free(row_buf);
  //cout << "Immagine corretta: " << prefix << endl;
}
#endif
