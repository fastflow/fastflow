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
 * Implementation of the Quicksort algorithm using FastFlow.
 *
 */

#include <stdio.h>
#include <ff/config.hpp>
#include <ff/farm.hpp>
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
    fprintf(stderr,"Usage: ff_qs <sz> <threshold> <nworkers>\n\n");
    fprintf(stderr,"       sz                : size of unsorted array\n");
    fprintf(stderr,"       bubble-threashold : threashold for sequential sorting\n");
    fprintf(stderr,"       nworkers          : the n. of FastFlow worker threads\n");
}

class my_loadbalancer: public ff_loadbalancer {
protected:
    // implement your policy...
    inline size_t selectworker() { return victim; }

public:
    // this is necessary because ff_loadbalancer has non default parameters....
    my_loadbalancer(int max_num_workers):ff_loadbalancer(max_num_workers) {}

    void set_victim(int v) { victim=v;}

private:
    size_t victim;
};


class Worker: public ff_node {
public:
    // int svc_init() {
    //     printf("Worker %d  is on core %d\n", get_my_id(), ff_getMyCpu());
    //     return 0;
    // }

    void * svc(void * t) {
        ff_task * task = (ff_task*)t;
        int i=task->i, j=task->j;

        do {
            if (j - i <= thresh) {
                std::sort(&A[i],&A[i]+((j-i)+1));
                //QuickSort(i,j);
                task->i = -1; // reset the value
                return task;
            } 
            int pivot = FindPivot(i,j);
            int k     = Partition(i,j,A[pivot]);
            
            task->i=k;
            ff_send_out(task);
            j = k-1;
            task = new ff_task(i,j);
        } while(1);

        return NULL;
    }
};

class Emitter: public ff_node {
public:
    Emitter(int nworkers, my_loadbalancer * const lb):streamlen(0),nworkers(nworkers),load(nworkers,0),lb(lb) {};

    // int svc_init() {
    //     printf("Emitter is on core %d\n", ff_getMyCpu());
    //     return 0;
    // }

    void * svc(void * t) {	
        ff_task * task = (ff_task*)t;

        // at the beginning we produce 2 tasks
        if (task == NULL) {
            int pivot = FindPivot(0,size-1);
            int k     = Partition(0,size-1,A[pivot]);
         
            task = new ff_task(0,k-1);
            load[0] += k-1;
            lb->set_victim(0);
            ff_send_out(task);
      
            task = new ff_task(k,size-1);
            load[1%nworkers] += size-1-k;
            lb->set_victim(1%nworkers);
            ff_send_out(task);
            
            streamlen=2;
            
            return GO_ON;       
        }
               
        if (task->i == -1) {
            --streamlen;
            delete task;
            if (streamlen == 0) return NULL;
            return GO_ON;
        }

        // schedule the new task to the less loaded worker
        std::vector<unsigned>::iterator idx_it = std::min_element(load.begin(),load.end());
        int idx = idx_it-load.begin();
        lb->set_victim(idx);
        load[idx] += (task->j-task->i);

        ff_send_out(task);        
        ++streamlen;
        
        return GO_ON;
    }
    
#if 0
    void svc_end() {
        std::cerr << "Printing load:\n";
        for(int i=0;i<nworkers;++i) 
            std::cerr << i << ": " << load[i] << "\n";
    }
#endif

private:
    unsigned int streamlen;
    int nworkers;
    std::vector<unsigned> load;
    my_loadbalancer * lb;
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

    /* The mapping policy is very simple: the emitter is mapped along on CPU 0, 
     * the workers on the remaining cores.
     */
    
    // the worker_mapping array, should be dinamically built
    const char worker_mapping[] = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31";

    threadMapper::instance()->setMappingList(worker_mapping);  

    ff_farm<my_loadbalancer> farm(false, nworkers*1024);    
    Emitter E(nworkers,farm.getlb());
    E.setAffinity(0); // the emitter is statically mapped on core 0

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


