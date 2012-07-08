#ifndef _SPD_CLUSTERS_HPP_
#define _SPD_CLUSTERS_HPP_

//  g++ salt_and_pepper_sort.cpp -lboost_program_options-mt -I /opt/local/include/ -L/opt/local/lib/ 

/*
  This approximate the cluctering of independent regions with stencil 
  (i-1,j)->(i,j) (i,j-1)->(i,j) on a bitmap

  It can be used to split a bitmap of noisy pixels in regions than can be
  indipendently restored using Concetto's Bayesian denoiser.

  Non-convex hulls are managed as they were convex, i.e. a false dependency is
  detected by the algorithm. The approximation can be eliminated by increasing   the complexity. This not probably worth as teh aproximation does not affect    correctness of next step.

  Complexity is O(M), where M is the number of noisy pixels in a NxN bitmap

  Author: Marco Aldinucci
  Date: Sunday, 14 Nov 2010
  State: proof-of-concept testing
   
*/

#include "definitions.h"
#include "utils.h"
#include "control_structures.hpp"
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <stack>
using namespace std;

/*
  template <class InputIterator1, class InputIterator2, class OutputIterator>
  OutputIterator mymerge ( InputIterator1 first1, InputIterator1 last1,
  InputIterator2 first2, InputIterator2 last2,
  OutputIterator result )
  {
  while (true) {
  *result++ = (*first2 < *first1)? *first2++ : *first1++;
  if (first1==last1) return copy(first2,last2,result);
  if (first2==last2) return copy(first1,last1,result);
  }
  }
*/

/*
  template<class T>
  T  merge_move(T& a,  T& b) {
  T ret;
  //copy(b.begin(), b.end(), back_inserter(ret));
  merge(a.begin(),a.end(),b.begin(),b.end(),back_inserter(ret));
  a.clear();
  b.clear();
  b.erase(b.begin(),b.end());
  a.erase(a.begin(),a.end());
  return ret;
  }
*/


template <typename T>
void clusters_check(vector<vector<noisy<T> > > &rows, int n) {
  long rowcount = 0;
  long elemcount = 0;
  vector<long> sizes;
  for (unsigned long k=0;k<rows.size();++k) {	
    if (rows[k].size()>0) {
      sizes.push_back(rows[k].size());
      elemcount += rows[k].size();
      //cout << "Count " << rows[k].size() << "\n";
      ++rowcount;
    }
  }
  cout << "N. of clusters after cleaning " << rowcount << endl;
  cout << "N. of elements after cleaning " << elemcount << endl;

  sort(sizes.begin(),sizes.end());
  for (unsigned long k=0;(k<sizes.size()) && (k<(unsigned int)n);++k)
    cout << "C-" << k << " = " << sizes[sizes.size()-k-1] << endl;
}

// ------------


/*
  void build_convex_clusters(vector<vector<vector<t_ind> > > &rows, t_ind *noisy, int n_noisy) {
  // let us assume they are stored in row-major order
  // find adjacent pixels in rows (can be parallalized on different rows)
  // at the moment of detection

  //compacting rows
  int cur_row = -2;
  int prev_col;
  int cur_set;
  for (int i=0; i<n_noisy; ++i) {
  if (noisy[i].r > cur_row) {
  vector<point> elem;
  elem.push_back(noisy[i]);
  vector<vector<point> > row;
  row.push_back(elem);
  cur_row = noisy[i].r;
  rows[cur_row] = row;
  prev_col = noisy[i].c;
  cur_set = 0;
  //cout << "New row " << noisy[i].r << "\n";
  } else {
  if (noisy[i].c == (prev_col+1)) {
  // merge cells
  (rows[cur_row][cur_set]).push_back(noisy[i]);
  ++prev_col;
  } else {
  vector<point> elem;
  elem.push_back(noisy[i]);
  rows[cur_row].push_back(elem);
  ++cur_set;
  prev_col= noisy[i].c;
  }
  }
  }
  
  //compacting columns
  
  int prev_set;
  for (int i = 1; i < rows.size(); ++i) {
  cur_set = 0; 
  prev_set = 0;
  while ( (prev_set < rows[i-1].size()) && (cur_set < rows[i].size()) &&
  (rows[i-1].size()>0) && rows[i].size()>0 ) {
  int right = MIN(rows[i][cur_set][rows[i][cur_set].size()-1].c,
  rows[i-1][prev_set][rows[i-1][prev_set].size()-1].c);
  int left = MAX(rows[i][cur_set][0].c,rows[i-1][prev_set][0].c);
  int min_left = MIN(rows[i][cur_set][0].c,rows[i-1][prev_set][0].c);
  if ( right >= left) {
  // merge sets (up to down)
  rows[i][cur_set] = 
  merge_move<vector<point> >(rows[i][cur_set],rows[i-1][prev_set]);
  ++prev_set;
  } else {
  if (min_left == rows[i][cur_set][0].c)
  ++cur_set;
  else
  ++prev_set;
  }
  }
  }
  }

  void build_clusters(vector<vector<vector<t_ind> > > &rows, t_ind *noisy, int n_noisy,int max_size) {
  // let us assume they are stored in row-major order
  // find adjacent pixels in rows (can be parallalized on different rows)
  // at the moment of detection

  //compacting rows
  int cur_row = -2;
  int prev_col;
  int cur_set;
  for (int i=0; i<n_noisy; i++) {
  if (noisy[i].r > cur_row) {
  vector<point> elem;
  elem.push_back(noisy[i]);
  vector<vector<point> > row;
  row.push_back(elem);
  cur_row = noisy[i].r;
  rows[cur_row] = row;
  prev_col = noisy[i].c;
  cur_set = 0;
  //cout << "New row " << noisy[i].r << "\n";
  } else {
  if (noisy[i].c == (prev_col+1)) {
  // merge cells
  (rows[cur_row][cur_set]).push_back(noisy[i]);
  prev_col++;
  } else {
  vector<point> elem;
  elem.push_back(noisy[i]);
  rows[cur_row].push_back(elem);
  cur_set++;
  prev_col= noisy[i].c;
  }
  }
  }
  
  //compacting columns v2
 
  int prev_set;
  for (int i = 1; i < rows.size(); i++) {
  cur_set = 0; 
  prev_set = 0;
  while ( (prev_set < rows[i-1].size()) && (cur_set < rows[i].size()) &&
  (rows[i-1].size()>0) && rows[i].size()>0 ) {
  bool notadj = true;
  point min = point();
  // point prev_min = point();
  for (int k=0; (k<rows[i][cur_set].size()) && (notadj);k++)
  for (int s=0; (s<rows[i-1][prev_set].size()) && (notadj);s++) {
  notadj = !((rows[i][cur_set][k].c==rows[i-1][prev_set][s].c) &&
  (rows[i][cur_set][k].r-1==rows[i-1][prev_set][s].r));
  if ((rows[i][cur_set][k].c < min.c) && 
  (rows[i][cur_set][k].r == i)) min = rows[i][cur_set][k];
  if ((rows[i-1][prev_set][s].c < min.c) && 
  (rows[i-1][prev_set][s].r == (i-1))) 
  min = rows[i-1][prev_set][s];
  }
  if (!notadj) {
  if (rows[i][cur_set].size()+rows[i-1][prev_set].size()<max_size)
  rows[i][cur_set] = 
  merge_move<vector<point> >(rows[i][cur_set],rows[i-1][prev_set]);
  else cout << "Cannot merge " << rows[i][cur_set].size() << " - " << rows[i-1][prev_set].size() << "\n";
  ++prev_set;
  } else {
  if (min.r == i)
  ++cur_set;
  else
  ++prev_set;
  }	  
  }
	
  if (rows[i].size()>1) {
  for (int h=0; h<rows[i].size()-1;h++) {
  bool notadj = true;
		
  for (int k=0; (k<rows[i][h].size()) && (notadj);k++)
  for (int s=0; (s<rows[i][h+1].size()) && (notadj);s++)
  notadj = !((rows[i][h][k].c==rows[i][h+1][s].c) &&
  ((rows[i][h][k].r-1==rows[i][h+1][s].r) ||
  (rows[i][h][k].r==rows[i][h+1][s].r-1)));
		
  if (!notadj) {
  //cout << "Match found\n";
  if (rows[i][h].size()+rows[i][h+1].size()<max_size)	
  rows[i][h] = merge_move<vector<point> >(rows[i][h],rows[i][h+1]);
  else cout << "Cannot merge2 " << rows[i][h].size() << " - " << rows[i][h+1].size() << "\n";
  }
		
  }
  }
  }
 
  }
*/


template <typename T>
void build_perfect_clusters(
			    vector<vector<T> > &rows,
			    T *noisy,
			    bmp_size_t n_noisy
			    )
{
  int cur_cluster=0;
  stack<int> todo;
  vector<pair<int,int> > merges;
  int ti;
  int gt = 0;
 
  vector<T> t;
  rows.push_back(t);
  for (bmp_size_t i=0; i<n_noisy; ++i) {
	//cout << noisy[i].r << " - " << noisy[i].c << endl;
    if (noisy[i].cluster < 0) {
      //not yet clustered
      todo.push(i);
      int pushcount =1;
      while (!todo.empty()) {
	ti = todo.top();
	todo.pop();
	if (noisy[ti].cluster < 0) {
	  //not yet clustered
	  noisy[ti].cluster = cur_cluster;
	  rows[cur_cluster].push_back(noisy[ti]);

	  if (noisy[ti].right_noisy) {
	    //right-dependent is noisy
	    //cout << "right is noisy\n";
	    if (noisy[noisy[ti].right_noisy_i].cluster<0) {
	      //right-dependent not yet clustered
	      //cout << "pushing "<< noisy[ti].right_noisy_i << endl;
	      todo.push(noisy[ti].right_noisy_i);
	      ++pushcount;
	    } else {
	      //right-dependent already clustered
		  
	      if (noisy[noisy[ti].right_noisy_i].cluster != cur_cluster) {
		//cout << "merging required" << cur_cluster << " -> " << noisy[noisy[ti].right_noisy_i].cluster << "\n";
		pair<int,int> m = pair<int,int>(noisy[noisy[ti].right_noisy_i].cluster, cur_cluster);
		merges.push_back(m);
	      }			  
	    } 
	  }
	  if (noisy[ti].up_noisy) {
	    //up-dependent is noisy
	    if (noisy[noisy[ti].up_noisy_i].cluster < 0) {
	      //up-dependent not yet clustered
	      todo.push(noisy[ti].up_noisy_i);
	      ++pushcount;
	    } else {
	      //up-dependent already clustered
	      if (noisy[noisy[ti].up_noisy_i].cluster != cur_cluster) {
		//cout << "merging required" << cur_cluster << " -> " << noisy[noisy[ti].up_noisy_i].cluster << "\n";
		pair<int,int> m = pair<int,int>(noisy[noisy[ti].up_noisy_i].cluster,cur_cluster);
		merges.push_back(m);
				
	      }
	    }
	  }
	} else
	  if (noisy[ti].cluster != cur_cluster)
	    cerr << "Something nasty happened in clustering  "<< noisy[ti].cluster << " c2 " << cur_cluster << "\n" ;
      }
      gt +=pushcount;
      // correctness check
      /*
	for (int k = 0; k < n_noisy; k++) {
	if ((noisy[k].up_noisy_i==i) && (noisy[k].cluster!=cur_cluster))
	cerr << "Error down\n";
	if ((noisy[k].right_noisy_i==i) && (noisy[k].cluster!=cur_cluster))
	cerr << "Error right\n";
	}
      */
      ++cur_cluster;
      rows.push_back(t);
    } 
  }
  


  
  //cout << "Clustered points " << gt << " N. of clusters " << rows.size() << " N. of required merges " << merges.size() << endl;
  /*
    int rowcount = 0;
    int elemcount = 0;
  */
  /*
  for (unsigned int k=0;k<rows.size();++k) {	
    if (rows[k].size()>0) {
      //elemcount += rows[k].size();
      //cout << "Count " << rows[k].size() << "\n";
      //++rowcount;
      sort(rows[k].begin(),rows[k].end());
    }
  }
  */
  /*
  cout << "N. of clusters before cleaning " << rowcount << endl;
  cout << "N. of elements before cleaning " << elemcount << endl;
  */

  // cleaning duplicates
  // transitive clousure
  sort(merges.begin(),merges.end());
  bool fixpoint = false;
  //int d,s1,s2;
  while (!fixpoint) {
    fixpoint = true;
    //s1=s2=0;
    for (unsigned int k=0;k<merges.size();++k) {
      //cout << "Merge " << k << endl;
      //rows[merges[k].first] = merge_move<vector<t_ind> >(rows[merges[k].first],rows[merges[k].second]);
      rows[merges[k].first].insert(rows[merges[k].first].end(), rows[merges[k].second].begin(), rows[merges[k].second].end());
      rows[merges[k].second].clear();
      for (unsigned int s=k+1;s<merges.size();++s) {
	if (((merges[s].second == merges[k].second) &&
	     (merges[s].first == merges[k].first)) || 
	    ((merges[s].second == merges[k].first) &&
	     (merges[s].first == merges[k].second))) {
	  merges.erase(merges.begin()+s);
	  //++d;
	  //cerr << "Removing duplicate merge\n";
	}
	else {
	  if (merges[s].second == merges[k].second) {
	    merges[s].second=merges[k].first;
	    fixpoint = false;
	    //++s1;
	    //cout << "Sub2\n";
	  }
		 
	  if (merges[s].first == merges[k].second) {
	    //cout << "Sub1\n";
	    merges[s].first = merges[k].first;
	    fixpoint = false;
	    //++s2;
	  }
	}
      }
    }
    //cout << "Iterate Duplicates " << d << " reduction1  "<<  s1  << " reduction2 "  << s2 << "\n";
  }

  for (unsigned int k=0;k<rows.size();++k) 
    sort(rows[k].begin(),rows[k].end());
 
  //clusters_check(rows,20);

   //clusters_check(noisy_sets,5);
  // resize clusters - par approximation per load balancing
  /*
    unsigned long max_partition = n_noisy/set_fraction;
    long initial_size = noisy_sets.size();
    for (int k=0; k<initial_size; ++k) {	
    while (noisy_sets[k].size()>max_partition){
    //int cs = noisy_sets.size();
    vector<noisy_t> t;
    noisy_sets.push_back(t);
    noisy_sets[noisy_sets.size()-1].insert(noisy_sets[noisy_sets.size()-1].end(),noisy_sets[k].end()-max_partition,noisy_sets[k].end());
    noisy_sets[k].erase(noisy_sets[k].end()-max_partition,noisy_sets[k].end());
    }
    }
  */
  //clusters_check(noisy_sets,5);

}

#endif
