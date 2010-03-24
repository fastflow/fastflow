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
 * Author: Massimo Torquati
 * March 2010
 *
 * Simple test for the unbounded SWSR buffer. It tests also the memory 
 * allocator.
 *
 */
#include <iostream>
#include <node.hpp>   // for Barrier
#include <deque>
#include <buffer.hpp>
#include <ubuffer.hpp>
#include <spin-lock.hpp>
#include <atomic/atomic.h>

using namespace ff;

#if defined(USE_FFA)
#include <allocator.hpp>
#if !defined(_FFA_)
static ff_allocator ffa;
#define malloc(x) ffa.malloc(x)
#define free(x)   ffa.free(x)
// it is possible to use also this one
//#define free(x)   (ff::FFAllocator::instance())->free(x)
#else // try FFAllocator
#define malloc(x) (ff::FFAllocator::instance())->malloc(x)
#define free(p)   (ff::FFAllocator::instance())->free(p)
#endif // _FFA_
#endif // USE_FFA
#if defined(USE_TBB)
#include <tbb/scalable_allocator.h>
#define malloc(x)   (scalable_malloc(x))
#define free(x)     (scalable_free(x))
#endif
#if !defined(USE_DEQUE)
#if defined(USE_SCALABLE_QUEUE)
#include <scalable_queue.h>
multi_queue* b;
#else
#if defined(FF_BOUNDED) // try circular SWSR buffer
SWSR_Ptr_Buffer   * b;
#else
uSWSR_Ptr_Buffer  * b;  // try unbounded SWSR buffer
#endif 
#endif /* SCALABLE_QUEUE */
#else   // try deque
#if defined(USE_SPINLOCK)
lock_t block;
#define LOCK(x) spin_lock(x)
#define UNLOCK(x) spin_unlock(x)
#else  // with mutexes
#include <pthread.h>
static pthread_mutex_t block = PTHREAD_MUTEX_INITIALIZER;
#define LOCK(x) pthread_mutex_lock(&x)
#define UNLOCK(x) pthread_mutex_unlock(&x)
#endif // USE_SPINLOCK
std::deque<void*> * b;
#endif

int ntasks=0; // the number of tasks

void * P(void *) {
#if defined(USE_FFA) && !defined(_FFA_)
  ffa.registerAllocator();
#endif

    Barrier::instance()->barrier();
    ffTime(START_TIME);
    
    for(int i=0;i<ntasks;++i) {
	int * p =(int*)malloc(8*sizeof(int));
	if (p) {
	    for(register int j=0;j<8;++j) p[j]=i+j;
#if !defined(USE_DEQUE)

#if defined(USE_SCALABLE_QUEUE)
	    int result;
	    do { 
		result = 
		    enqueue_multi( b, (void*) p);
	    } while ( result == 0 );
	    
#else
	    bool result;
	    do { result = b->push(p); } while(!result);
#endif  /* SCALABLE_QUEUE */


#else
	    LOCK(block);
	    b->push_back(p);
	    UNLOCK(block);
#endif
	}
    }
#if !defined(USE_DEQUE)
#if defined(USE_SCALABLE_QUEUE)
    int result;
    do { 
	result = 
	    enqueue_multi( b, (void*) FF_EOS);
    } while ( result == 0 );
#else
    bool result; 
    do { result = b->push((void*)FF_EOS); } while(!result);
#endif

#else
    LOCK(block);
    b->push_back((void*)FF_EOS);
    UNLOCK(block);
#endif
    pthread_exit(NULL);
}
    

void * C(void *) {
#if defined(USE_FFA) && !defined(_FFA_)
    ffa.register4free();
#endif

    Barrier::instance()->barrier();

    bool end=false;
    void * task=NULL;
    int k=0;
    while(!end) {
    retry:
#if !defined(USE_DEQUE)
#if defined(USE_SCALABLE_QUEUE)
	int result;
	do { 
	    result = 
		dequeue_multi( b, &task);
	} while ( result == 0 );
	if (1) {
#else
	if (b->pop(&task)) {
#endif
#else // USE_DEQUE
	LOCK(block);
	if (b->size()) {
	    task = (b->front();
	    b->pop_front();
	    UNLOCK(block);
#endif

	
	    if (task == (void*)FF_EOS) { 
		end=true;
	    } else {
        for(register int j=0;j<8;++j) {
            int * t = (int *)task;
		    if (t[j]!=(k+j)) {
			std::cout << " ERROR, value is " << t[j] << " should be " << k+j << "\n";
		    } else t[j]++; // just write in the array
        }
		++k;
		
		free(task);
	    }
	} else {
#if defined(USE_DEQUE)
	    UNLOCK(block);
#endif 
	    goto retry;
	}
    }
    ffTime(STOP_TIME);	
    pthread_exit(NULL);
}



int main(int argc, char * argv[]) {
    if (argc!=3) {
	std::cerr << "use: " << argv[0] << " (base-)buffer-size ntasks\n";
	return -1;
    }

    int  size = atoi(argv[1]);
    ntasks= atoi(argv[2]);

#if defined(USE_FFA) && !defined(_FFA_)
    int nslabs[N_SLABBUFFER] = {16384,0,0,0,0,0,0,0,0};
    ffa.init(nslabs);
#endif

    pthread_t P_handle, C_handle;
#if !defined(USE_DEQUE)

#if defined(USE_SCALABLE_QUEUE)
    /* create the multi queue */
    b = create_multi_queue( size );
    
#else // !USE_SCALABLE_QUEUE
#if defined(FF_BOUNDED)
    b = new SWSR_Ptr_Buffer(size);
#else
    b = new uSWSR_Ptr_Buffer(size);
#endif
    if (b->init()<0) abort();
#endif // USE_SCALABLE_QUEUE

#else // USE_DEQUE
    b = new std::deque<void*>;
#endif

    Barrier::instance()->barrier(2);

    ffTime(START_TIME);

    if (pthread_create(&P_handle, NULL,P,NULL) != 0) {
	abort();
    }

    if (pthread_create(&C_handle, NULL,C,NULL) != 0) {
	abort();
    }

    pthread_join(C_handle,NULL);
    pthread_join(P_handle,NULL);


    ffTime(STOP_TIME);
    std::cerr << "DONE, time= " << ffTime(GET_TIME) << " (ms)\n";
#if !defined(USE_DEQUE)
    delete b;
#endif

#if defined(USE_FFA) && defined(ALLOCATOR_STATS)
#if defined(_FFA_)
    (ff::FFAllocator::instance())->printstats();
#else
    ffa.printstats();
#endif
#endif

    return 0;
}
