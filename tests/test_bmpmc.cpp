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
/*
 * Simple test for the bounded Multi-Producer/Multi-Consumer MSqueue.
 *
 * Author: Massimo Torquati
 *  April 2012
 *
 */

#include <cstdio>
#include <vector>
#include <algorithm>
//#include <pthread.h>
#include <ff/platforms/platform.h>
#include <ff/atomic/atomic.h>
#include <ff/atomic/abstraction_dcas.h>
#include <ff/MPMCqueues.hpp>
#include <ff/node.hpp>


int    NTHREADS;
size_t MYSIZE;
atomic_long_t counter;           
ff::MPMC_Ptr_Queue* q=NULL;
ff::Barrier *bar = NULL;


// consumer function
void * consumer(void * arg) {
    int myid= *(int*)arg;
    size_t * data;

    bar->doBarrier(myid);
    while(1) {
	if (q->pop((void**)&data)) {
	    printf("(%d %ld) ", myid, (long)data);
	    atomic_long_inc(&counter);
	}
	if ((size_t)(atomic_long_read(&counter))>= MYSIZE) break;
    }
    pthread_exit(NULL);
    return NULL;
}

int main(int argc, char* argv[]) {
    long qs = 20;
    int nc = 6;

    if (argc>1) {
        if (argc!=3) {
            std::cerr << "use: "
                      << argv[0]
                      << " queue-size #consumers\n";
            return -1;
        }
        qs=atol(argv[1]);
        nc=atoi(argv[2]);
    }


    MYSIZE= qs;
    assert(MYSIZE>0);
    NTHREADS=nc;
    assert(NTHREADS>0);

    q = new ff::MPMC_Ptr_Queue;
    assert(q);
    q->init(MYSIZE);

    for(size_t i=1;i<=MYSIZE;++i) 
        q->push((void*)i);

    atomic_long_set(&counter,0);

    pthread_t * C_handle;

    C_handle = (pthread_t *) malloc(sizeof(pthread_t)*NTHREADS);
	
    // define the number of threads that are going to partecipate....
    bar = new ff::Barrier;
    bar->barrierSetup(NTHREADS);

    int * idC;
    idC = (int *) malloc(sizeof(int)*NTHREADS);
    for(int i=0;i<NTHREADS;++i) {
        idC[i]=i;
        if (pthread_create(&C_handle[i], NULL,consumer,&idC[i]) != 0) {
            abort();
        }
    }

    // wait all consumers
    for(int i=0;i<NTHREADS;++i) {
        pthread_join(C_handle[i],NULL);
    }
    printf("\n");
    return 0;

}
