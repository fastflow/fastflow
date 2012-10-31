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
 * Blocking implementation of the Quicksort algorithm using MPMC. 
 *
 */

#include <stdio.h>
#include <pthread.h>
#include <ff/farm.hpp>
#include <ff/MPMCqueues.hpp>
#include <algorithm>  // it is needed to use std::sort

using namespace ff;

//FastFlow task type
struct ff_task {
    ff_task():i(-1),j(-1) {};
    ff_task(int i,int j):i(i),j(j) {};
    int i;
    int j;
};

// some globals 
unsigned size    = 0;
int thresh       = 0;
int nworkers     = 0;
bool done        = false;
unsigned int   *A= NULL;
pthread_mutex_t    mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t     cond  = PTHREAD_COND_INITIALIZER;
unsigned int       numtasks = 0;
int                numwaiting = 0;
MSqueue          * msq = NULL;

static inline bool QPUSH(void * task) {
    do ; while(!(msq->push(task)));
    return true;
}   
#define QPOP(x) msq->pop(x)


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
    if (j-i <= 1) {
        if (A[i]>A[j]) std::swap(A[i],A[j]);
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
        std::swap(A[i], A[(random() % (size-i)) + i]);
    
}


void usage() {
    fprintf(stderr,"Usage: ff_qs_mpmc <sz> <threshold> <nworkers>\n\n");
    fprintf(stderr,"       sz                : size of unsorted array\n");
    fprintf(stderr,"       bubble-threashold : threashold for sequential sorting\n");
    fprintf(stderr,"       nworkers          : the n. of FastFlow worker threads\n");
}



// generic thread worker 
class Worker: public ff_node {
public:
    void * svc(void *) {
        while(1) {
            ff_task * task;
            
            while (!done && !QPOP((void**)&task)) {
                pthread_mutex_lock(&mutex);
                if (numwaiting==(nworkers-1)) {                        
                    pthread_cond_broadcast(&cond);
                    pthread_mutex_unlock(&mutex);
                    done = true;
                    break;
                }
                ++numwaiting;
                pthread_cond_wait(&cond, &mutex);
                --numwaiting;
                pthread_mutex_unlock(&mutex);                        
            }

            if (done) break; // work finished
            
            int i=task->i, j=task->j;

            //printf("thread %d got (%d, %d)\n", get_my_id(), i, j);
            do { 
                if (j - i <= thresh) {
                    std::sort(&A[i],&A[i]+((j-i)+1));
                    //QuickSort(i,j);
                    delete task;
                    --numtasks;
                    break; // try to get work from the queue
                } 
                int pivot = FindPivot(i,j);
                int k   = Partition(i,j,A[pivot]);
                
                // produce one task
                ff_task * task2 = new ff_task(k,j);
                QPUSH(task2);

                pthread_mutex_lock(&mutex);
                if (numwaiting) pthread_cond_signal(&cond);
                ++numtasks;
                pthread_mutex_unlock(&mutex);
                
                j=(k-1);
            } while(1);
        }
        return NULL;
    }
};


// it just produces the first 2 tasks in the MPMC queue and than exits
class Emitter: public ff_node {
public:
    void * svc(void *) {	
        int pivot = FindPivot(0,size-1);
        int k     = Partition(0,size-1,A[pivot]);
         
        ff_task * task = new ff_task(0,k-1);
        QPUSH(task);

        task = new ff_task(k,size-1);
        QPUSH(task);
        
        numtasks = 2;
            
        // start all workers
        for(int i=0;i<nworkers;++i)
            ff_send_out(GO_ON);
        
        return NULL;      
    }
};


int main(int argc, char *argv[]) {
    bool check_result=false;

    if (argc<4 || argc>5) {
        usage();
        return -1;
    } 
    
    size   = atoi(argv[1]);
    thresh = atoi(argv[2]);
    nworkers=atoi(argv[3]);
    if (argc==5) check_result=true;
    
    if (nworkers > ff::ff_farm<>::DEF_MAX_NUM_WORKERS) {
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
    
    // initialize MPMC queue
    msq = new MSqueue;
    assert(msq->init());

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
                error("wrong result, A[%d]=%d (correct value is %d)\n",i,A[i],i);
                //print_array();
                return -1;
            }
        printf("Ok\n");
    }

    delete [] A;
    return 0;
}


