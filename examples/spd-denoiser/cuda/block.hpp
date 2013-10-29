#ifndef _SPD_BLOCK_HPP_
#define _SPD_BLOCK_HPP_
#include "definitions.h"
#include "control_structures.hpp"
#include <vector>
#include <iterator>
//#include <iostream>
#include <algorithm>
using namespace std;

template <typename Tnoisy>
void build_blocks(
		  vector<vector<Tnoisy> > &noisy_sets,  // dst
		  vector<vector<Tnoisy> > &noisy_sets_, // src
		  unsigned int n_noisy,
		  unsigned int n_blocks
		  )
{
  unsigned int step = (n_noisy % n_blocks==0) ? n_noisy/n_blocks: 1+n_noisy/n_blocks;
  //block copy
  unsigned long long n_stored = 0;
  unsigned int i_block = 0;
  unsigned int i_block_ = 0;
  //dest 
  noisy_sets[0].reserve(step);
  //typename vector<Tnoisy>::iterator start_block = noisy_sets[0].begin();
  long long free_elem_in_dest = step;
  //source
  typename vector<Tnoisy>::const_iterator start_block_ = noisy_sets_[0].begin();
  typename vector<Tnoisy>::const_iterator end_block_ = noisy_sets_[0].end();
  long long to_be_copied_in_src = end_block_-start_block_;
  while(n_stored < n_noisy) {
	   if (free_elem_in_dest==0) {
		  ++i_block;
		  free_elem_in_dest=step;
		  noisy_sets[i_block].reserve(step);
	  }
	  if (to_be_copied_in_src==0) {
		  ++i_block_;
		  start_block_ = noisy_sets_[i_block_].begin();
		  end_block_ = noisy_sets_[i_block_].end();
		  to_be_copied_in_src=end_block_-start_block_;
	  }
	  long long to_be_copied = (std::min)(free_elem_in_dest, to_be_copied_in_src);
	  copy(start_block_,start_block_+to_be_copied, back_inserter(noisy_sets[i_block]));
	  start_block_+=to_be_copied;
//	  copy(start_block_,end_block_, back_inserter(noisy_sets[i_block]));
	  free_elem_in_dest-=to_be_copied;
	  n_stored+=to_be_copied;
	  to_be_copied_in_src-=to_be_copied;	 
  }
}



/*
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
*/
#endif
