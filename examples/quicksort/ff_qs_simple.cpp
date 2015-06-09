/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \link
 *  \file ff_qs_simple.cpp
 *  \ingroup application_level
 *
 *  \brief TODO
 *
 */

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
 * Implementation of the Quicksort algorithm using FastFlow.
 *
 */

#include <stdio.h>
#include <ff/config.hpp>
#include <ff/farm.hpp>

using namespace ff;

//FastFlow task type
typedef struct {
    int i;
    int j;
    int k;
} ff_task;

// some globals 
unsigned size    = 0;
int thresh       = 0;
unsigned int   *A= NULL;


void print_array() {
  int j=0;
  
  for(unsigned int i=0;i<size;i++) {
      if (j == 15) {
	  printf("\n");
	  j=0;
      }
      j++;
      printf("%d ", A[i]);
  }
  printf("\n\n");
}


inline void swap(int i, int j) {
  register int tmp;
  tmp = A[i]; A[i] = A[j]; A[j] = tmp;
}


/* Return the largest of first two keys. */
inline int FindPivot(int i, int j) {
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
        swap(left,right);
        while (A[left]  <  pivot) left++;
        while (A[right] >= pivot) right--;
    } while (left <= right);
    
    return(left);
}


inline void QuickSort(int i, int j) {
    if (j-i <= 1) {
	if (A[i]>A[j]) swap(i,j);
	return;
    } 
    int pivot = FindPivot(i,j);
    int k     = Partition(i,j,A[pivot]);
    QuickSort(i, k-1);
    QuickSort(k,j);
}


void initArray() {
    /* All of the elements are unique. */
    for (unsigned int i = 0; i < size; i++)	A[i] = i;
    
    /* Shuffle them randomly. */
    srandom(0);
    for (unsigned int i = 0; i < size; i++)	
	swap(i, (random() % (size-i)) + i);
    
}


void usage() {
    fprintf(stderr,"Usage: ff_qs <sz> <threshold> <nworkers>\n\n");
    fprintf(stderr,"       sz                : size of unsorted array\n");
    fprintf(stderr,"       bubble-threashold : threashold for sequential sorting\n");
    fprintf(stderr,"       nworkers          : the n. of FastFlow worker threads\n");
}


class Worker: public ff_node {
public:
    void * svc(void * t) {
        ff_task * task = (ff_task*)t;
        int i=task->i, j=task->j;
        
        if (j - i <= thresh) {
            QuickSort(i,j);
            task->k = -1; // reset the value
            return task;
        } 
        int pivot = FindPivot(i,j);
        task->k   = Partition(i,j,A[pivot]);
        
        return task;
    }
};

class Emitter: public ff_node {
public:
    Emitter():streamlen(0) {};

    void * svc(void * t) {	
        ff_task * task = (ff_task*)t;
        if (task == NULL) {
            int pivot = FindPivot(0,size-1);
            int k     = Partition(0,size-1,A[pivot]);
         
            task = new ff_task;
            task->i=0; task->j=k-1;
            ff_send_out(task);
      
            task = new ff_task;
            task->i=k; task->j=size-1;
            ff_send_out(task);
            
            streamlen=2;
            
            return GO_ON;       
        }
        
        
        int i=task->i, j=task->j, k=task->k;
        --streamlen;
        if (k==-1) {
            if (streamlen == 0) {
                delete task;
                return NULL;
            }
            delete task;
            return GO_ON;
        }
        
        task->i=i; task->j=k-1;	task->k=-1;
        ff_send_out(task);
        
        task = new ff_task;
        task->i=k; task->j=j; task->k=-1;
        ff_send_out(task);
        
        streamlen +=2;
        
        return GO_ON;
    }
private:
    unsigned int streamlen;
};


int main(int argc, char *argv[]) {
    bool check_result=false;

    if (argc<4 || argc>5) {
        usage();
        return -1;
    } 
    
    size   = atoi(argv[1]);
    thresh = atoi(argv[2]);
    int nworkers=atoi(argv[3]);
    if (argc==5) check_result=true;
    
    if (nworkers > DEF_MAX_NUM_WORKERS) {
        fprintf(stderr, "too many number of workers\n");
        return -1;
    }
    if ((unsigned)thresh > (size/2)) { // just a simple check
        fprintf(stderr, "threshold too high\n");
        return -1;        
    }
    
    A = new unsigned int[size];
    if (!A) {
        fprintf(stderr,"Not enough memory for A\n");
        exit(1);
    }
    
    initArray();
    
    ff_farm<> farm(false, nworkers*1024);    
    Emitter E;
    farm.add_emitter(&E);
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);
    farm.wrap_around();
    
    printf("starting....\n");
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    printf("Time: %g (ms)\n", farm.ffTime());
    
    if (0) print_array();

    if (check_result) {
        for(unsigned int i=0;i<size;i++) 
            if (A[i]!=i) {
                error("wrong result\n");
                return -1;
            }
        printf("Ok\n");
    }

    delete [] A;
    return 0;
}

/*!
 *  @}
 *  \endlink
 */
