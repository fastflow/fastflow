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
#include <pthread.h>
#include <ff/atomic/atomic.h>
#include <ff/atomic/abstraction_dcas.h>
#include <ff/MPMCqueues.hpp>
#include <ff/node.hpp>


int NTHREADS;
int SIZE;
atomic_long_t counter;           
ff::MPMC_Ptr_Queue* q=NULL;

// consumer function
void * consumer(void * arg) {
    int myid= *(int*)arg;
    int * data;

    ff::Barrier::instance()->doBarrier();
    while(1) {
	if (q->pop((void**)&data)) {
	    printf("(%d %ld) ", myid, (long)data);
	    atomic_long_inc(&counter);
	}
	if ((long)(atomic_long_read(&counter))>= SIZE) break;
    }
    pthread_exit(NULL);
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
	std::cerr << "use: " << argv[0] 
		  << " queue-size #consumers\n";
	return -1;
    }

    SIZE= atoi(argv[1]);
    assert(SIZE>0);
    NTHREADS=atoi(argv[2]);
    assert(NTHREADS>0);

    q = new ff::MPMC_Ptr_Queue(SIZE);
    assert(q);

    for(int i=1;i<=SIZE;++i) 
	q->push((void*)i);
    
    atomic_long_set(&counter,0);

    pthread_t * C_handle;

    C_handle = (pthread_t *) malloc(sizeof(pthread_t)*NTHREADS);
	
    // define the number of threads that are going to partecipate....
    ff::Barrier::instance()->doBarrier(NTHREADS);

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
