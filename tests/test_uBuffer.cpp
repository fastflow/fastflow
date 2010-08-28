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
 *
 * Simple test for the unbounded SWSR buffer. 
 * It tests also the memory allocator and bounded buffer.
 *
 */
#include <iostream>
#include <deque>
#include <ff/node.hpp>   // for Barrier
#include <ff/buffer.hpp>
#include <ff/ubuffer.hpp>
#include <ff/dynqueue.hpp>
#include <ff/spin-lock.hpp>
#include <ff/atomic/atomic.h>
#include <ff/cycle.h>

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
#if defined(FF_BOUNDED) // try circular SWSR buffer
SWSR_Ptr_Buffer   * b;
#else
#if defined(FF_DYNAMIC)
dynqueue * b;
#else  
uSWSR_Ptr_Buffer  * b;  // try unbounded SWSR buffer
#endif // FF_DYNAMIC
#endif // FF_BOUNDED
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

int WARMUP=0;           // number of tasks for the warm-up
int ntasks=0;            // total number of tasks
int cpu_P=-1, cpu_C=-1;  // cpu's number
#if defined(PAPI_PERF)
static int EventSet = PAPI_NULL;
#endif


#if defined(__linux__)

#include <asm/unistd.h>
#define gettid() syscall(__NR_gettid)
#include <sys/resource.h>

static inline const int numCores() {
    FILE       *f;
    int         n;
    
    f = popen("cat /proc/cpuinfo |grep processor | wc -l", "r");
    fscanf(f, "%d", &n);
    pclose(f);
    return n;
}

static inline unsigned long getCpuFreq() {
    FILE       *f;
    unsigned long long t;
    float       mhz;

    f = popen("cat /proc/cpuinfo |grep MHz |head -1|sed 's/^.*: //'", "r");
    fscanf(f, "%f", &mhz);
    t = (unsigned long)(mhz * 1000000);
    pclose(f);
    return (t);
}


/*
 * priority_level is a value in the range -20 to 19.
 * The default priority is 0, lower priorities cause more favorable scheduling.
 *
 */
static inline int mapThreadToCpu(int cpu_id, int priority_level=0) {
#ifdef CPU_SET
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    if (sched_setaffinity(gettid(), sizeof(mask), &mask) != 0) 
        return -1;

    if (priority_level) 
        if (setpriority(PRIO_PROCESS, gettid(),priority_level) != 0) {
            perror("setpriority:");
            return -2;
        }
#else
#warning "CPU_SET not defined, cannot map thread to specific CPU"
#endif

    return 0;
}

#else // __linux__

static inline int mapThreadToCpu(int cpu_id, int priority_level=0) {
    return -1;
}
#endif



static inline void PUSH(const int i) {
#if defined(NO_ALLOC) 
    int * p = (int *)(0x1234+i);
    if (1) {
#else       
    int * p = (int*)malloc(8*sizeof(int));
    if (p) {
        for(register int j=0;j<8;++j) p[j]=i+j;
#endif

#if !defined(USE_DEQUE)

#if defined(FF_BOUNDED) && defined(MULTIPUSH)
        static void * data[32];
        static int cdata=0;
        data[cdata++]=p;
        if (cdata==32) {
            do ; while(!( b->multipush(data,32)));
            cdata=0;
        }
#else
	    do ; while(!(b->push(p)));
#endif

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
        if (mapThreadToCpu(cpu_P)!=0)
            std::cerr << "Cannot map producer thread to cpu " << cpu_P << ", going on...\n";

    for(int i=0;i<WARMUP;++i) PUSH(i);

    Barrier::instance()->barrier();
    ffTime(START_TIME);

    for(int i=WARMUP;i<ntasks;++i) PUSH(i);
 
#if !defined(USE_DEQUE)
    bool result; 
    do { result = b->push((void*)FF_EOS); } while(!result);
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



    bool end=false;
    void * task=NULL;
    int k=0;
#if defined(PAPI_PERF)
    int retval; 
	if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK ) {
        std::cerr << "PAPI_create_eventset " << retval << "\n";
        exit(-1);
    }

    const int event = PAPI_L2_DCM;
    if ( PAPI_add_event( EventSet, event ) != PAPI_OK ) {
        std::cerr << "PAPI_add_event error\n";
        exit(-1);
    }


    if ( (retval=PAPI_start( EventSet ))  != PAPI_OK ) {
        std::cerr << "PAPI_start error\n";
        perror("PAPI_start");
        std::cerr << "retval= " << retval << "\n";
        exit(-1);
    }
#endif // PAPI_PERF


    if (cpu_C != -1) 
        if (mapThreadToCpu(cpu_C)!=0)
            std::cerr << "Cannot map consumer thread to cpu " << cpu_C << ", going on...\n";

    Barrier::instance()->barrier();
   
    while(!end) {
    retry:
#if !defined(USE_DEQUE)
	if (b->pop(&task)) {
#else 
	LOCK(block);
	if (b->size()) {
	    task = b->front();
	    b->pop_front();
	    UNLOCK(block);
#endif

	
	    if (task == (void*)FF_EOS) { 
		end=true;
	    } else {
#if defined(NO_ALLOC)
            if (task != (void *)(0x1234+k))
                std::cerr << " ERROR, wrong task received\n";
#else
        for(register int j=0;j<8;++j) {
            int * t = (int *)task;
		    if (t[j]!=(k+j)) {
			std::cerr << " ERROR, value is " << t[j] << " should be " << k+j << "\n";
		    } else t[j]++; // just write in the array
        }
#endif
		++k;

#if !defined(NO_ALLOC)		
		free(task);
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
    if ( PAPI_stop( EventSet, NULL )  != PAPI_OK )
        std::cerr << "PAPI_stop error\n";
    
    long long int value=0;
    if ( ( retval = PAPI_read( EventSet, &value ) ) != PAPI_OK )
        std::cerr << "PAPI_read error\n";
    else
        std::cout << "L2 miss: " << value << "\n";
#endif  

    pthread_exit(NULL);
}



int main(int argc, char * argv[]) {
    if (argc!=3) {
        if (argc == 5) {
            cpu_P = atoi(argv[3]);
            cpu_C = atoi(argv[4]);
#if defined(__linux__)
            int nc = numCores();
            if (cpu_P < 0 || cpu_P >= nc) {
                std::cerr << "Wrong producer thread CPU number, range is [0-" << nc << "[\n";
                return -1;
            }
            if (cpu_C < 0 || cpu_C >= nc) {
                std::cerr << "Wrong consumer thread CPU number, range is [0-" << nc << "[\n";
                return -1;
            }
#endif            
            ;
        } else {        
            std::cerr << "use: " << argv[0] << " (base-)buffer-size ntasks [#P-core] [#C-core]\n";
            return -1;
        }
    }
    int  size = atoi(argv[1]);
    ntasks= atoi(argv[2]);


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

#if defined(FF_BOUNDED)
    b = new SWSR_Ptr_Buffer(size);
    if (b->init()<0) abort();
#else
#if defined(FF_DYNAMIC)
    b = new dynqueue(size,true);
#else
    b = new uSWSR_Ptr_Buffer(size);
    if (b->init()<0) abort();
#endif // FF_DYNAMIC
#endif

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
