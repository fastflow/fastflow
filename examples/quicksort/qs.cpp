/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/* Author: Massimo Torquati <torquati@di.unipi.it> <massimotor@gmail.com>
 *
 * Implementation of the Quicksort algorithm. 
 * Quicksort is a recursive sorting algorithm that operates by repeatedly 
 * partitioning an unsorted input list into a pair of unsorted sub-lists, 
 * such that all of the elements in one of the sub-lists are strictly greater 
 * than the elements of the other, and then recursively invoking itself on 
 * the two unsorted sub-lists.
 * 
 * This is just a naive implementation of the algorithm, without any specific 
 * optimisation. More efficient implementations are possible.
 *
 */


#include <stdlib.h>
#include <stdio.h>
#include <ff/utils.hpp>
#include <algorithm>  // to use std::sort

unsigned int  size=0;            // array size
unsigned int *A=NULL;  // array which needs to be ordered


void print_array() {
    int j=0;
    for(unsigned  i=0;i<size;i++) {
        if (j == 15) {
            printf("\n");
            j=0;
        }
        j++;
        printf("%d ", A[i]);
    }
    printf("\n\n");
}


//inline void swap(int i, int j) {
//  register int tmp;  
//  tmp = A[i]; A[i] = A[j]; A[j] = tmp;
//}

/* Return the largest of first two keys. */
inline int FindPivot(int i, int j) {
    // NOTE: this is not the best choice for the pivot.
    register int pivot = (i+(j-i))/2;

    if (A[i]>A[pivot]) {
        if (A[i]>A[j]) return i;
        else return j;
    } else {
        if (A[pivot]>A[j]) return pivot;
        else return j;
    }
    
    // if (A[i] > A[i+1]) return i;
    //  else  return i+1;
}
			

/* Partition the array between elements i and j around the specified pivot. */
inline int Partition(int i, int j, unsigned int pivot) {
    int left = i;
    int right = j;
  
    do {
        std::swap(A[left],A[right]);
        while (A[left]  <  pivot) left++;
        while (A[right] >= pivot) right--;
    } while (left <= right);
    
    return(left);
}


		
inline void QuickSort(int i, int j) {
    if (j-i < 2) {
        if (A[i]>A[j]) std::swap(A[i],A[j]);
        return;
    } 
    int pivot = FindPivot(i,j);
    int k     = Partition(i,j,A[pivot]);
    QuickSort(i, k-1);
    QuickSort(k,j);
}


// initialise the array A with unique elementes
void initArray() {
  unsigned int i;

  for (i = 0; i < size; i++) A[i] = i;
  
  /* Shuffle them randomly. */
  srandom(0);
  for (i = 0; i < size; i++)	
      std::swap(A[i], A[(random() % (size-i)) + i]);
  
}



int main(int argc, char * argv[]) {
    bool check_result=false;
    if (argc<2 || argc>3) {
        fprintf(stderr, "use: %s size\n", argv[0]);
        return -1;
    }
    
    size = atoi(argv[1]);
    if (argc==3) check_result=true;

    A = new unsigned int[size];
    if (!A) {
        fprintf(stdout,"Not enough memory for A\n");
        exit(1);
    }
    
    initArray();    
    printf("starting....\n");
    ff::ffTime(ff::START_TIME);
    QuickSort(0, size-1); // my own threashold based Quicksort algo
    //std::sort(A,A+size);
    ff::ffTime(ff::STOP_TIME);
    printf("Time: %g (ms)\n", ff::ffTime(ff::GET_TIME));
    
    if (0) print_array();

    if (check_result) {
        for(unsigned int i=0;i<size;i++) 
            if (A[i]!=i) {
                fprintf(stderr,"wrong result, A[%d]=%d (correct value is %d)\n",i,A[i],i);
                return -1;
            }
        printf("Ok\n");
    }
    
    delete [] A;
    return 0;
}


