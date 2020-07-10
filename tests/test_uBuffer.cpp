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
 *  March   2010
 *  August  2010 (improved)
 *  October 2010 (added mpush)
 * 
 * Simple test for the unbounded SWSR buffer. 
 * It tests also the memory allocator and bounded buffer.
 *
 */
#include <iostream>
#include <deque>
#include <ff/node.hpp>   // for Barrier
#include <ff/dynqueue.hpp>
#include <ff/buffer.hpp>
#include <ff/ubuffer.hpp>
#include <ff/staticlinkedlist.hpp>
#include <ff/spin-lock.hpp>
#include <ff/cycle.h>
#include <ff/mapping_utils.hpp>

#if defined(PAPI_PERF)
#include "papiStdEventDefs.h"
#include <papi.h>
#endif

using namespace ff;

// allocator defines

#if !defined(NO_ALLOC)
#if defined(USE_FFA)
#include <ff/allocator.hpp>
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
#endif // NO_ALLOC

// queue
#if !defined(USE_DEQUE)
#if defined(FF_MPMC)
#include <ff/MPMCqueues.hpp>
MSqueue * b;
#else
#if defined(FF_STATIC_LINKED_LIST)
staticlinkedlist * b;
#else
#if defined(FF_BOUNDED) // try circular SWSR buffer
#if defined(LAMPORT)
Lamport_Buffer    * b;
#else
SWSR_Ptr_Buffer   * b;
#endif
#else
#if defined(FF_DYNAMIC)
dynqueue * b;
#else  
uSWSR_Ptr_Buffer  * b;  // try unbounded SWSR buffer
#endif // FF_DYNAMIC
#endif // FF_BOUNDED
#endif // FF_STATIC_LINKED_LIST
#endif // FF_MPMC
#else  // try deque
#if defined(USE_SPINLOCK) // with spin-lock
lock_t block;
#define LOCK(x) spin_lock(x)
#define UNLOCK(x) spin_unlock(x)
#else                    // with mutexes
#include <pthread.h>
static pthread_mutex_t block = PTHREAD_MUTEX_INITIALIZER;
#define LOCK(x) pthread_mutex_lock(&x)
#define UNLOCK(x) pthread_mutex_unlock(&x)
#endif // USE_SPINLOCK
std::deque<void*> * b;
#endif // USE_DEQUE

int WARMUP=0;            // number of tasks for the warm-up
int size=0;              // buffer size
int ntasks=0;            // total number of tasks
int cpu_P=-1, cpu_C=-1;  // cpu's number
Barrier *bar = NULL;
#if defined(PAPI_PERF)
static int EventSet = PAPI_NULL;
#endif

#if defined(uSWSR_MULTIPUSH) || defined(SWSR_MULTIPUSH)
#define MULTIPUSH 1
#endif

#if defined(COMPUTES)
#include <math.h>
inline double f(double x) { return sin(x); }
inline double g(double x) { return cos(x); }
#endif

#if defined(COMPUTES)
static union {double a; void *b;} x;
static double y=0.65432012;
#endif

static inline void PUSH(const int i) {
#if defined(NO_ALLOC) 
    long * p = (long *)(0x1234+i);
    if (1) {
#else       
    long * p = (long *)malloc(8*sizeof(long));

    if (p) {
        for(int j=0;j<8;++j) p[j]=i+j;
#endif

        //ticks_wait(0); // just to slow down a bit the producer
#if defined(COMPUTES)
        x.a = 3.1415*f(x.a);
        p= (long *)(x.b);
#endif

#if !defined(USE_DEQUE)
 #if defined(TWO_LOCK)
       do {} while(!(b->mp_push(p)));
 #else

  #if defined(MULTIPUSH)
        do {} while(!(b->mpush(p)));
  #else 
	    do {} while(!(b->push(p)));
  #endif
 #endif // TWO_LOCK
#else // USE_DEQUE

	    LOCK(block);
	    b->push_back(p);
	    UNLOCK(block);
#endif
	}
}   


void * P(void *) {
#if defined(USE_FFA) && !defined(_FFA_)
  ffa.registerAllocator();
#endif


#if defined(PAPI_PERF)    
	if ( PAPI_register_thread() != PAPI_OK )
        std::cerr << "PAPI_register_thread\n";
#endif

    if (cpu_P != -1) 
        if (ff_mapThreadToCpu(cpu_P)!=0)
            std::cerr << "Cannot map producer thread to cpu " << cpu_P << ", going on...\n";

#if defined(COMPUTES)
    x.a=0.12345678;
#endif
    for(int i=0;i<WARMUP;++i) PUSH(i);

    bar->doBarrier(0);
    ffTime(START_TIME);

    for(int i=WARMUP;i<ntasks;++i) PUSH(i);
 
#if !defined(USE_DEQUE)
    bool result; 
    do { 
#if defined(MULTIPUSH)
        result = b->mpush((void*)FF_EOS); 
#else        
        result = b->push((void*)FF_EOS); 
#endif
    } while(!result);
#else
    LOCK(block);
    b->push_back((void*)FF_EOS);
    UNLOCK(block);
#endif

#if defined(MULTIPUSH)
    while(!b->flush());
#endif

    pthread_exit(NULL);
	return NULL;
}
    

void * C(void *) {
#if defined(UBUFFER_STATS)
    const int STATS_SAMPLE=1000;
    std::vector<unsigned long> length, miss,hit;
    length.reserve(ntasks/STATS_SAMPLE +1);
    miss.reserve(ntasks/STATS_SAMPLE +1);
    hit.reserve(ntasks/STATS_SAMPLE +1);
#endif
#if defined(USE_FFA) && !defined(_FFA_)
    ffa.register4free();
#endif

    bool end=false;
    union {double a; void *b;} task;
    int k=0;

    if (cpu_C != -1) 
        if (ff_mapThreadToCpu(cpu_C)!=0)
            std::cerr << "Cannot map consumer thread to cpu " << cpu_C << ", going on...\n";
    
#if defined(PAPI_PERF)
    int retval; 
	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK ) {
        std::cerr << "PAPI_create_eventset " << retval << "\n";
        exit(-1);
    }

    const int eventlist[] = {
		PAPI_L1_DCM,
		PAPI_L2_DCM,
		0
	};

    int i=0;
    while(eventlist[i]!=0) {
        if ( PAPI_add_event( EventSet, eventlist[i] ) != PAPI_OK ) {
            std::cerr << "PAPI_add_event error " << i << "\n";
            exit(-1);
        }
        ++i;
    }

    if ( (retval=PAPI_start( EventSet ))  != PAPI_OK ) {
        std::cerr << "PAPI_start error\n";
        perror("PAPI_start");
        std::cerr << "retval= " << retval << "\n";
        exit(-1);
    }
#endif // PAPI_PERF

    bar->doBarrier(1);
   
    while(!end) {
    retry:
#if !defined(USE_DEQUE)
  #if defined(TWO_LOCK)
    if (b->mp_pop(&task.b)) {
  #else
	if (b->pop(&task.b)) {
  #endif // TWO_LOCK
#else 
	LOCK(block);
	if (b->size()) {
	    task.b = b->front();
	    b->pop_front();
	    UNLOCK(block);
#endif
	
	    if (task.b == (void*)FF_EOS) { 
            end=true;
	    } else {
#if defined(COMPUTES)
            y+= task.a - g(y);
#else
#if defined(NO_ALLOC)
            if (task.b != (void *)(0x1234+k))
                std::cerr << " ERROR, wrong task received " << task.b << " expected " << (void *)(0x1234+k) << " \n";
#else
        for(int j=0;j<8;++j) {
            long * t = (long *)task.b;
		    if (t[j]!=(k+j)) {
			std::cerr << " ERROR, value is " << t[j] << " should be " << k+j << "\n";
		    } else t[j]++; // just write in the array
        }
#endif
#endif

		++k;

#if defined(UBUFFER_STATS)
        if ((k%STATS_SAMPLE)==0){
            length.push_back(b->queue_status());
            miss.push_back(b->readMiss());
            hit.push_back(b->readHit());
        }
#endif

#if !defined(NO_ALLOC)		
		free(task.b);
#endif
	    }
	} else {
#if defined(USE_DEQUE)
	    UNLOCK(block);
#endif 
	    goto retry;
	}
    }
    ffTime(STOP_TIME);	

#if defined(PAPI_PERF)
    long long int values[4];
    char descr[PAPI_MAX_STR_LEN];
    
    if ( PAPI_stop( EventSet, values )  != PAPI_OK )
        std::cerr << "PAPI_stop error event " << i << "\n";

    i=0;
    while(eventlist[i]!=0) {
        PAPI_event_code_to_name( eventlist[i], descr );
        std::cout << descr << " " << values[i] << "\n";
        ++i;
    }
#endif  

#if defined(UBUFFER_STATS)
    printf("\nlength:\n");
    for(unsigned int i=0;i<length.size();++i)
        printf("%ld ", length[i]);
    printf("\nmiss:\n");
    for(unsigned int i=0;i<miss.size();++i)
        printf("%ld ", miss[i]);
    printf("\nhit:\n");
    for(unsigned int i=0;i<hit.size();++i)
        printf("%ld ", hit[i]);
#endif

#if defined(COMPUTES)
    printf("result y=%f\n", y);
#endif

    pthread_exit(NULL);
	return NULL;
}



int main(int argc, char * argv[]) {
    std::cout << "Num of cores " <<  ff_numCores() << "\n";
    std::cout << "Frequency " <<  ff_getCpuFreq() << "\n";
    size=1024;
    ntasks=1000000;
    if (argc>1) {
        if (argc!=3) {
            if (argc == 5) {
                cpu_P = atoi(argv[3]);
                cpu_C = atoi(argv[4]);
                int nc = ff_numCores();
                if (cpu_P < 0 || cpu_P >= nc) {
                    std::cerr << "Wrong producer thread CPU number, range is [0 - " << nc << "[\n";
                    return -1;
                }
                if (cpu_C < 0 || cpu_C >= nc) {
                    std::cerr << "Wrong consumer thread CPU number, range is [0 - " << nc << "[\n";
                    return -1;
                }
            } else {        
                std::cerr << "use: " << argv[0] << " (base-)buffer-size ntasks [#P-core] [#C-core]\n";
                return -1;
            }
        }
        size = atoi(argv[1]);
        ntasks= atoi(argv[2]);
    }

#if defined(COMPUTES)
    if (sizeof(double) != sizeof(long)) {
        std::cerr << "cannot execute the \"COMPUTES\" code on this architecture\n";
        return -1;
    }
#endif


#if defined(PAPI_PERF)
    const PAPI_hw_info_t *hwinfo = NULL;
    int retval;
    if ( ( retval =
		   PAPI_library_init( PAPI_VER_CURRENT ) ) != PAPI_VER_CURRENT ) {
        std::cerr << "PAPI_library_init " << retval << "\n";
        exit(-1);
    }

    if ( ( retval =
		   PAPI_thread_init( ( unsigned
							   long ( * )( void ) ) ( pthread_self ) ) ) !=
		 PAPI_OK ) {
        std::cerr << "PAPI_thread_init error\n";
        exit(-1);
	}

	if ( ( hwinfo = PAPI_get_hardware_info(  ) ) == NULL ) {
        std::cerr << "PAPI_get_hardware_info\n";
        exit(-1);
	}



	/* Extract and report the cache information */
	PAPI_mh_level_t * L = ( PAPI_mh_level_t * ) ( hwinfo->mem_hierarchy.level );
	for ( int i = 0; i < hwinfo->mem_hierarchy.levels; i++ ) {
		for ( int j = 0; j < 2; j++ ) {
			int tmp;

			tmp = PAPI_MH_CACHE_TYPE( L[i].cache[j].type );
			if ( tmp == PAPI_MH_TYPE_UNIFIED ) {
				printf( "L%d Unified ", i + 1 );
			} else if ( tmp == PAPI_MH_TYPE_DATA ) {
				printf( "L%d Data ", i + 1 );
			} else if ( tmp == PAPI_MH_TYPE_INST ) {
				printf( "L%d Instruction ", i + 1 );
			} else if ( tmp == PAPI_MH_TYPE_VECTOR ) {
				printf( "L%d Vector ", i + 1 );
			} else if ( tmp == PAPI_MH_TYPE_TRACE ) {
				printf( "L%d Trace ", i + 1 );
			} else if ( tmp == PAPI_MH_TYPE_EMPTY ) {
				break;
			} else {
                std::cerr << "PAPI_get_hardware_info \n";
			}

			tmp = PAPI_MH_CACHE_WRITE_POLICY( L[i].cache[j].type );
			if ( tmp == PAPI_MH_TYPE_WB ) {
				printf( "Write back " );
			} else if ( tmp == PAPI_MH_TYPE_WT ) {
				printf( "Write through " );
			} else {
                std::cerr << "PAPI_get_hardware_info\n";
			}

			tmp = PAPI_MH_CACHE_REPLACEMENT_POLICY( L[i].cache[j].type );
			if ( tmp == PAPI_MH_TYPE_PSEUDO_LRU ) {
				printf( "Pseudo LRU policy " );
			} else if ( tmp == PAPI_MH_TYPE_LRU ) {
				printf( "LRU policy " );
			} else if ( tmp == PAPI_MH_TYPE_UNKNOWN ) {
				printf( "Unknown policy " );
			} else {
                std::cerr << "PAPI_get_hardware_info\n";
			}

			printf( "Cache:\n" );
			if ( L[i].cache[j].type ) {
				printf
					( "  Total size: %dKB\n  Line size: %dB\n  Number of Lines: %d\n  Associativity: %d\n\n",
					  ( L[i].cache[j].size ) >> 10, L[i].cache[j].line_size,
					  L[i].cache[j].num_lines, L[i].cache[j].associativity );
			}
		}
	}

#endif

    
#if defined(USE_FFA) && !defined(_FFA_)
    int nslabs[N_SLABBUFFER] = {16384,0,0,0,0,0,0,0,0};
    ffa.init(nslabs);
#endif

    pthread_t P_handle, C_handle;
#if !defined(USE_DEQUE)
#if defined(FF_MPMC)
    b = new MSqueue;
    if (!b) abort();
    if (!b->init()) abort();
#else
#if defined(FF_STATIC_LINKED_LIST)
    b = new staticlinkedlist(size, true);
#else
#if defined(FF_BOUNDED)
#if defined(LAMPORT)
    b = new Lamport_Buffer(size);
    if (b->init()<0) abort();
#else
    b = new SWSR_Ptr_Buffer(size);
    if (b->init()<0) abort();
#endif //LAMPORT
#else
#if defined(FF_DYNAMIC)
    b = new dynqueue(size,true);
#else
    b = new uSWSR_Ptr_Buffer(size);
    if (!b->init()) abort();
#endif // FF_DYNAMIC
#endif // FF_BOUNDED 
#endif // FF_STATIC_LINKED_LIST
#endif // FF_MPMC

#else // USE_DEQUE
    b = new std::deque<void*>;
#endif

    bar = new Barrier;
    bar->barrierSetup(2);

    ffTime(START_TIME);

    if (pthread_create(&P_handle, NULL,P,NULL) != 0) {
	abort();
    }

    if (pthread_create(&C_handle, NULL,C,NULL) != 0) {
	abort();
    }

  
    pthread_join(C_handle,NULL);
    pthread_join(P_handle,NULL);


    std::cout << "DONE, time= " << ffTime(GET_TIME) << " (ms)\n";
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
