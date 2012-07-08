#ifndef _SPD_BLOCK_HPP_
#define _SPD_BLOCK_HPP_
#include "definitions.h"
#include "control_structures.hpp"
#include <vector>
//#include <iostream>
#include <algorithm>
using namespace std;

template <typename Tnoisy>
void build_blocks(
		  vector<vector<Tnoisy> > &noisy_sets,
		  vector<vector<Tnoisy> > &noisy_sets_,
		  unsigned int n_noisy,
		  unsigned int n_blocks
		  )
{
  unsigned int step = (n_noisy % n_blocks==0) ? n_noisy/n_blocks: 1+n_noisy/n_blocks;
  //block copy
  unsigned int n_stored = 0;
  unsigned int i_block = 0;
  unsigned int i_block_ = 0;
  //dest
  typename vector<Tnoisy>::const_iterator start_block = noisy_sets[0].begin();
  typename vector<Tnoisy>::const_iterator end_block = start_block + step;
  noisy_sets[0].reserve(step);
  //source
  typename vector<Tnoisy>::const_iterator start_block_ = noisy_sets_[0].begin();
  typename vector<Tnoisy>::const_iterator end_block_ = noisy_sets_[0].end();
  while(n_stored < n_noisy) {
    unsigned int free_block = end_block - start_block;
    unsigned int unstored_block_ = end_block_ - start_block_;
    if(unstored_block_ <= free_block) {
      //store the whole source block
      copy(start_block_, end_block_, back_inserter(noisy_sets[i_block]));
      start_block += unstored_block_;
      n_stored += unstored_block_;
      //change source block
      ++i_block_;
      start_block_ = noisy_sets_[i_block_].begin();
      end_block_ = noisy_sets_[i_block_].end();
    }
    else {
      //partially store the source block
      copy(start_block_, start_block_ + free_block, back_inserter(noisy_sets[i_block]));
      //cerr << "ok copy" << endl;
      start_block_ += free_block;
      n_stored += free_block;
      //change dest block
      ++i_block;
      start_block = noisy_sets[i_block].begin();
      end_block = start_block + step;
      noisy_sets[i_block].reserve(step);
    }
  }
}



#ifdef BORDER
template <typename T>
void find_borders(
		  vector<pair<unsigned int, unsigned int> > &borders,
		  vector<vector<noisy<T> > > &noisy_sets,
		  unsigned int n_blocks,
		  bmp_size_t height, bmp_size_t width,
		  Bitmap<> &bmp
		  ) {
  borders.reserve(n_blocks * width);
  for (unsigned int i=0; i< noisy_sets.size(); ++i) 
    for (unsigned int j=0; j< noisy_sets[i].size(); ++j)
      bmp.set_block_index(noisy_sets[i][j].c,noisy_sets[i][j].r,i);
  unsigned int r,c;
  for (int i=0; i< (int) noisy_sets.size(); ++i)
    for (unsigned int j=0; j< noisy_sets[i].size(); ++j) {
      c = noisy_sets[i][j].c;
      r = noisy_sets[i][j].r;
      //cerr << "C " << c << " R " << r << "\n";
      if ( ((c < (width-1)) && bmp.get_noisy(c+1,r) && 
	    (bmp.get_block_index(c+1,r)!=i)) ||
	   ((c > 0) && bmp.get_noisy(c-1,r) && 
	    (bmp.get_block_index(c-1,r)!=i)) ||
	   ((r < (height-1)) && bmp.get_noisy(c,r+1) && 
	    (bmp.get_block_index(c,r+1)!=i)) ||
	   ((r > 0) && bmp.get_noisy(c,r-1) && 
	    (bmp.get_block_index(c,r-1)!=i)) )
	borders.push_back(pair<unsigned int,unsigned int>(i,j));  
    }
  //cerr << "Halo points " << borders.size() << "\n";
}
#endif
#endif
