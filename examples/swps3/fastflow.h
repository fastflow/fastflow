#ifndef __FF_TASK_
#define __FF_TASK_

#include <ff/allocator.hpp>

enum { MIN_ALLOC=4096, STEP_ALLOC=2048 };

#define getdbname(task) task->db
#define getdb(task)     task->db
#define getdata(task)   task->dbdata
#define getlen(task)    task->dbLen

typedef struct {
    int    dbLen;
    char * dbdata;

    char * db;
    size_t size;
    
    char * query;
    int    querylen;
    
    double score;     
} task_t;

#if !defined(USE_LIBC_ALLOCATOR)
#define FF_ALLOCATOR 1
static ff::ff_allocator * allocator = 0;
#define MALLOC(size)         (allocator->malloc(size))
#define FREE(ptr)            (allocator->free(ptr))
#define REALLOC(ptr,newsize) (allocator->realloc(ptr,newsize))
#define ALLOCATOR_INIT1() {						\
    int nslabs[N_SLABBUFFER] = {0,0,0,0,0,512,512,128,32};		\
	allocator=new ff_allocator();					\
	allocator->init(nslabs,false);					\
    }
#define ALLOCATOR_INIT() {						\
	allocator=new ff_allocator();					\
	allocator->init();						\
    }
#define ALLOCATOR_T ff::ff_allocator
#define ALLOCATOR (allocator)
#define ALLOCATOR_REGISTER      (allocator->registerAllocator())
#define ALLOCATOR_REGISTER4FREE (allocator->register4free())

#if defined(ALLOCATOR_STATS)
#define ALLOCATOR_ST            allocator->printstats(std::cerr)
#else
#define ALLOCATOR_ST
#endif

#else 

/* standard libc malloc/free */
#define MALLOC(size)            malloc(size)
#define FREE(ptr)               free(ptr)
#define REALLOC(ptr,newsize)    realloc(ptr,newsize)
#define ALLOCATOR_INIT()
#define ALLOCATOR_INIT1()
#define ALLOCATOR_T             void
#define ALLOCATOR               NULL
#define ALLOCATOR_REGISTER      (false)
#define ALLOCATOR_REGISTER4FREE (false)
#define ALLOCATOR_ST
#endif /* USE_LIBC_ALLOCATOR */


#endif /* __FF_TASK_ */
