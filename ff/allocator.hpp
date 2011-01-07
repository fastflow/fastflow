/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_ALLOCATOR_HPP_
#define _FF_ALLOCATOR_HPP_
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

/* Lock-free FastFlow allocator.
 *
 * Here we defined the ff_allocator (1) and the FFAllocator (2). 
 *
 * 1. The ff_allocator allocates only large chunks of memory, slicing them up 
 *    into little chunks all with the same size. Only one thread can perform 
 *    malloc operations while any number of threads may perform frees using 
 *    the ff_allocator. 
 * 
 *    The ff_allocator is based on the idea of Slab Allocator, for more details 
 *    about Slab Allocator please see:
 *     Bonwick, Jeff. "The Slab Allocator: An Object-Caching Kernel Memory 
 *     Allocator." Boston USENIX Proceedings, 1994.
 *
 * 2. Based on the ff_allocator, the FFAllocator has been implemented.
 *    It might be used by any number of threads to dynamically allocate/deallocate 
 *    memory. 
 *
 * Note: In all the cases it is possible, it is better to use the ff_allocator
 *       as it allows more control and is more efficient.
 *
 *  
 *
 */

/* - June 2008 first version
 *   Author: Massimo Torquati <torquati@di.unipi.it> <massimotor@gmail.com>
 *    
 */


#include <assert.h>
#include <algorithm>

#if defined(HAVE_ATOMIC_H)
#include <asm/atomic.h>
#else
#include <ff/atomic/atomic.h>
#endif

#if defined(__APPLE__)
#include <Availability.h>
#endif


#include <pthread.h>
#include <ff/cycle.h>
#include <ff/buffer.hpp>
#include <ff/ubuffer.hpp>
#include <ff/spin-lock.hpp>
#include <ff/svector.hpp>
#include <ff/utils.hpp>

//#define DEBUG_ALLOCATOR 1
#if defined(DEBUG_ALLOCATOR)
#define DBG(X) X
#else
#define DBG(X)
#endif

namespace ff {

enum { N_SLABBUFFER=9, MAX_SLABBUFFER_SIZE=8192};
static const int POW2_MIN  =5;
static const int POW2_MAX  = 13;
static const int buffersize[N_SLABBUFFER]     ={ 32, 64,128,256,512,1024,2048,4096,8192};
static const int nslabs_default[N_SLABBUFFER] ={512,512,512,512,128,  64,  32,  16,   8};

#if defined(ALLOCATOR_STATS)
struct all_stats {
    atomic_long_t      nmalloc;
    atomic_long_t      nfree;
    atomic_long_t      freeok1;
    atomic_long_t      freeok2;
    atomic_long_t      nrealloc;
    atomic_long_t      sysmalloc;
    atomic_long_t      sysfree;
    atomic_long_t      sysrealloc;
    atomic_long_t      hit;
    atomic_long_t      miss;
    atomic_long_t      nleak;
    atomic_long_t      leakremoved;
    atomic_long_t      memallocated;
    atomic_long_t      memfreed;
    atomic_long_t      segmalloc;
    atomic_long_t      segfree;
    atomic_long_t      newallocator;    
    atomic_long_t      deleteallocator;


    all_stats() {
        atomic_long_set(&nmalloc, 0);
        atomic_long_set(&nfree, 0);
        atomic_long_set(&freeok1, 0);
        atomic_long_set(&freeok2, 0);
        atomic_long_set(&nrealloc, 0);
        atomic_long_set(&sysmalloc, 0);
        atomic_long_set(&sysfree, 0);
        atomic_long_set(&sysrealloc, 0);
        atomic_long_set(&hit, 0);
        atomic_long_set(&miss, 0);
        atomic_long_set(&nleak, 0);
        atomic_long_set(&leakremoved, 0);
        atomic_long_set(&memallocated, 0);
        atomic_long_set(&memfreed, 0);
        atomic_long_set(&segmalloc, 0);
        atomic_long_set(&segfree, 0);
        atomic_long_set(&newallocator, 0);
        atomic_long_set(&deleteallocator, 0);
    }
    
    static all_stats *instance() {
        static all_stats allstats;
        return &allstats;
    }


    void print(std::ostream & out = std::cout) {
        out << "\n--- Allocator Stats ---\n" 
            << "malloc        = " << (unsigned long)atomic_long_read(&nmalloc)     << "\n"
            << "  cache hit   = " << (unsigned long)atomic_long_read(&hit)         << "\n"
            << "  cache miss  = " << (unsigned long)atomic_long_read(&miss)        << "\n"
            << "  leakremoved = " << (unsigned long)atomic_long_read(&leakremoved) << "\n\n"
            << "free          = " << (unsigned long)atomic_long_read(&nfree)       << "\n"
            << "  ok1         = " << (unsigned long)atomic_long_read(&freeok1)      << "\n"
            << "  ok2         = " << (unsigned long)atomic_long_read(&freeok2)      << "\n"
            << "  leakinsert  = " << (unsigned long)atomic_long_read(&nleak)       << "\n"
            << "realloc       = " << (unsigned long)atomic_long_read(&nrealloc)    << "\n\n"
            << "mem. allocated= " << (unsigned long)atomic_long_read(&memallocated)<< "\n"
            << "mem. freed    = " << (unsigned long)atomic_long_read(&memfreed)    << "\n\n"
            << "segmalloc     = " << (unsigned long)atomic_long_read(&segmalloc)   << "\n"
            << "segfree       = " << (unsigned long)atomic_long_read(&segfree)     << "\n\n"
            << "sysmalloc     = " << (unsigned long)atomic_long_read(&sysmalloc)   << "\n"
            << "sysfree       = " << (unsigned long)atomic_long_read(&sysfree)     << "\n"
            << "sysrealloc    = " << (unsigned long)atomic_long_read(&sysrealloc)  << "\n\n";
        if ((unsigned long)atomic_long_read(&newallocator)>0) {
            out << "\n ---- How many allocators ----\n"
                << "new allocator    = " << (unsigned long)atomic_long_read(&newallocator) << "\n"
                << "delete allocator = " << (unsigned long)atomic_long_read(&deleteallocator) << "\n\n";
        }
    }
};

#define ALLSTATS(X) X
#else
#define ALLSTATS(X)
#endif /* ALLOCATOR_STATS */


static inline bool isPowerOfTwoMultiple(size_t alignment, size_t multiple) {
    return alignment && ((alignment & (alignment-multiple))==0);
}
    


/*
 * This simple allocator uses standard libc allocator to 
 * allocate large chunk of memory.
 *
 */

class SegmentAllocator {
public:
    SegmentAllocator():memory_allocated(0) {}
    ~SegmentAllocator() {}

    void * newsegment(size_t segment_size) { 
        //void * ptr = ::malloc(segment_size);
        void * ptr=NULL;

#if __MAC_OS_X_VERSION_MIN_REQUIRED < __MAC_10_6
        ptr = ::malloc(segment_size);
#else
        if (posix_memalign(&ptr,sysconf(_SC_PAGESIZE),segment_size)!=0)
            return NULL; 
#endif
        memory_allocated += segment_size;

        ALLSTATS(atomic_long_inc(&all_stats::instance()->segmalloc); atomic_long_add(segment_size, &all_stats::instance()->memallocated));
        return ptr;
    }

    void   freesegment(void * ptr, size_t segment_size) { 
        DBG(assert(memory_allocated >=segment_size));
        ::free(ptr); 
        memory_allocated -= segment_size;
        ALLSTATS(atomic_long_inc(&all_stats::instance()->segfree); atomic_long_add(segment_size, &all_stats::instance()->memfreed));
    }
    
    const size_t getallocated() const { return memory_allocated; }
    
private:
    size_t       memory_allocated;
};
    
    
    // forward declarations
class SlabCache;
class ff_allocator;
    
/* Each slab buffers has a Buf_ctl data structure at the beginning of the data. */
struct Buf_ctl {  SlabCache  * ptr; };

/* Each slab segment has a Seg_ctl data structure at the beginning of the data. */
struct Seg_ctl {
    // number of buffers in use
    DBG(size_t     refcount);          
    // reference to the SlabCache that owns the segment
    SlabCache    * cacheentry;   
    // reference to the allocator
    ff_allocator * allocator;    
    // number of items in the buffer freelist
    //atomic_long_t        availbuffers; 
    unsigned long  availbuffers;
};


/* per thread data */
struct xThreadData {
    enum { MIN_BUFFER_ITEMS=8192, LEAK_CHUNK=4096 };
    
    xThreadData(const bool allocator, int nslabs):leak(0) { 
#if defined(FF_BOUNDED_BUFFER)
        b=0;
        if (!allocator) {
            b = (SWSR_Ptr_Buffer*)::malloc(sizeof(SWSR_Ptr_Buffer));
            if (!b) abort();
            new (b) SWSR_Ptr_Buffer((MIN_BUFFER_ITEMS>(2*nslabs))?MIN_BUFFER_ITEMS:(2*nslabs)); 
            b->init();
        }
#endif
        leak = (uSWSR_Ptr_Buffer*)::malloc(sizeof(uSWSR_Ptr_Buffer));
        if (!leak) abort();
        new (leak) uSWSR_Ptr_Buffer(LEAK_CHUNK); 
        leak->init();
    }

    ~xThreadData() { 
#if defined(FF_BOUNDED_BUFFER)
        if (b) { b->~SWSR_Ptr_Buffer(); ::free(b);}
#endif
        if (leak) { leak->~uSWSR_Ptr_Buffer(); ::free(leak);}
    }
    
#if defined(FF_BOUNDED_BUFFER)
    SWSR_Ptr_Buffer       * b;
    uSWSR_Ptr_Buffer      * leak;
    long padding[longxCacheLine-2];
#else
    uSWSR_Ptr_Buffer      * leak;
    long padding[longxCacheLine-1];
#endif
};

    
/*
 * Cache of slab segment. Each segment consists of contiguous memory
 * carved up into equal size chunks.
 *
 *      SlabCache
 *    ----------------- <---
 *   | size            |    |
 *   | cur ptr seg     |    |
 *   | availbuffers    |    |
 *   | vect of buf-ptr |    |
 *    -----------------     |
 *                          |      -------------- <-------------------<-----------
 *                          |     |             |                    |
 *                          |     v             |                    |
 *      (Slab segment)      |    ----------------------------------------------------
 *                          |   |              | p |               | p | 
 *                          |---|- cacheentry  | t |      data     | t |      data 
 *  <---------------------------|- allocator   | r |               | r |
 *   (ptr to ff_allocator)      |  availbuffers|   |               |   | 
 *                               ----------------------------------------------------
 *                              |    Seg_ctl   | ^ |      
 *                                               |                 
 *                                               |--- Buf_ctl  
 *
 *
 *  NOTE: the segment overhead is:  sizeof(Seg_ctl) + (nbuffer * BUFFER_OVERHEAD) 
 *
 */
class SlabCache {    
private:    
    enum {TICKS_TO_WAIT=500,TICKS_CNT=3};
    enum {BUFFER_OVERHEAD=sizeof(Buf_ctl)};
    enum {MIN_FB_CAPACITY=32};

    
    inline Seg_ctl * const getsegctl(Buf_ctl * buf) { return *((Seg_ctl **)buf); }

    inline int newslab() {
        void * ptr = alloc->newsegment(size*nslabs + sizeof(Seg_ctl) + nslabs*BUFFER_OVERHEAD);
        
        if (ptr) {
            Seg_ctl * seg      = (Seg_ctl *)ptr;
            DBG(seg->refcount      = 0);
            seg->cacheentry    = this;
            seg->allocator     = mainalloc;
            //atomic_long_set(&seg->availbuffers,nslabs);
            seg->availbuffers = nslabs;
            ++seg;
            *(Seg_ctl **)seg = (Seg_ctl *)ptr;
            buffptr          = (Buf_ctl *)seg; 
            availbuffers    += nslabs;
            seglist.push_back(ptr);
            return 0;
        } 

        return -1;
    }

    inline void freeslab(void * ptr, bool seglist_rem=true) { 
        alloc->freesegment(ptr, size*nslabs +  sizeof(Seg_ctl) + nslabs*BUFFER_OVERHEAD); 

        /* remove ptr from seglist */
        if (seglist_rem) {
            svector<void *>::iterator b(seglist.begin()), e(seglist.end());
            for(;b!=e;++b) 
                if ((*b)==ptr) {
                    seglist.erase(b);
                    return;
                }
            DBG(assert(1==0));
        }
    }

    inline void * getfrom_fb() {
        void * buf = 0;
#if defined(FF_BOUNDED_BUFFER)
        register SWSR_Ptr_Buffer * b = 0;
        for(register unsigned i=0;i<fb_size;++i) {
            DBG(assert(fb[i]));
            b = fb[i]->b;            
            if (b) {
                if (b->pop(&buf))   {
                    DBG(assert(buf));
                    return buf;
                }
            } 
        }
#endif
        // fallback:
        // try to pop something from the leak queue 
        // of each threads
        for(register unsigned i=0;i<fb_size;++i) {
            if (fb[i]->leak->pop((void **)&buf)) {
                ALLSTATS(atomic_long_inc(&all_stats::instance()->leakremoved));
                return buf;
            }
        }
        
        // the cache is empty
        return 0;
    }
    
    inline void checkReclaim(Seg_ctl  * seg) {
        if (++(seg->availbuffers) == (unsigned long)nslabs) 
            freeslab(seg); /* reclaim allocated memory */
    }


    inline ticks ticks_wait(ticks t1) {
        register ticks delta, t0 = getticks();
        do { delta = getticks()-t0; } while (delta < t1);
        return delta-t1;
    }

public:    
    SlabCache(ff_allocator * const mainalloc, 
              SegmentAllocator * const alloc, size_t sz, int ns):
        size(sz),nslabs(ns),fb(0),fb_capacity(0),fb_size(0),buffptr(0),availbuffers(0),
        alloc(alloc),mainalloc(mainalloc) {}
    
    ~SlabCache() { 
        if (!nslabs) return;

        svector<void *>::iterator b(seglist.begin()), e(seglist.end());
        for(;b!=e;++b) {
            freeslab(*b,false);
        }
        seglist.clear();
        if (fb) {
            for(unsigned i=0;i<fb_size;++i)
                if (fb[i]) {
                    fb[i]->~xThreadData();
                    ::free(fb[i]);
                }
            ::free(fb);
        }
        pthread_key_delete(fb_key);
    }

    int init(bool prealloc=true) {
        if (!nslabs) return 0;

        atomic_long_set(&nomoremalloc,0);

        init_unlocked(lock);

        fb = (xThreadData**)::malloc(MIN_FB_CAPACITY*sizeof(xThreadData*));
        if (!fb) return -1;
        fb_capacity = MIN_FB_CAPACITY;

        if (pthread_key_create(&fb_key,NULL)!=0) {
            error("ff_allocator::SlabCache: ops! no more keys available!!\n");
            return -1;
        }

        if (prealloc && (newslab()<0)) return -1;
        return 0;
    }
    
    inline xThreadData * register4free(const bool allocator=false) {
        DBG(assert(nslabs>0));
        xThreadData * xtd = (xThreadData*)pthread_getspecific(fb_key);
        if (!xtd) {
            xtd = (xThreadData*)::malloc(sizeof(xThreadData));
            if (!xtd) return NULL;
            new (xtd) xThreadData(allocator, nslabs);

            spin_lock(lock); 
            if (fb_size==fb_capacity) {
                xThreadData ** fb_new = (xThreadData**)::realloc(fb,(fb_capacity+MIN_FB_CAPACITY)*sizeof(xThreadData*));
                if (!fb_new) {
                    spin_unlock(lock);
                    return NULL; 
                }
                fb = fb_new;
                fb_capacity += MIN_FB_CAPACITY;
            }
            fb[fb_size++] = xtd;
            if (pthread_setspecific(fb_key, xtd)!=0) return NULL;
            spin_unlock(lock);
        }
        return xtd;
    }

    inline void deregisterAllocator(const bool reclaim) {
        DBG(assert(nslabs>0));
        atomic_long_set(&nomoremalloc,1);

        // try to reclaim some memory
        for(register unsigned i=0;i<fb_size;++i) {
            DBG(assert(fb[i]));
#if defined(FF_BOUNDED_BUFFER)
            void * buf = 0;
            SWSR_Ptr_Buffer * b = fb[i]->b;
            if (b) {
                if (reclaim) {
                    while(b->pop((void **)&buf)) { 
                        DBG(assert(buf));
                        Seg_ctl  * seg = *(Seg_ctl **)buf;
                        checkReclaim(seg);
                        ALLSTATS(atomic_long_inc(&all_stats::instance()->leakremoved));
                    }
                    union { Buf_ctl * buf2; void * ptr;} b;
                    while(fb[i]->leak->pop(&b.ptr)) { 
                        checkReclaim(getsegctl(b.buf2));                        
                        ALLSTATS(atomic_long_inc(&all_stats::instance()->leakremoved));
                    }
                } else  {
                    /* NOTE: free at least some buffer rooms 
                     * to avoid race conditions for buffer full 
                     */
                    for(register int i=0;i<5;++i) b->pop((void**)&buf);
                }
            } else 
#endif
            {
                if (reclaim) {
                    union { Buf_ctl * buf2; void * ptr; } b;
                    while(fb[i]->leak->pop(&b.ptr)) {
                        checkReclaim(getsegctl(b.buf2));                        
                    }
                }
            }
        }
    }

    /* get an item from slab */
    inline void * getitem() { 
        DBG(assert(nslabs>0));
        void * item = 0;

        /* try to get one item from the available ones */
        if (availbuffers) {     
        avail:
            Seg_ctl * seg = *((Seg_ctl **)buffptr);
            DBG(assert(seg));
            DBG(++seg->refcount);
            --(seg->availbuffers);

            DBG(if (seg->refcount==1) assert(availbuffers==nslabs));
            
            item = (char *)buffptr + BUFFER_OVERHEAD;   // set data pointer            
            if (--availbuffers) {            
                Seg_ctl ** ctl   = (Seg_ctl **)((char *)item + size);
                *ctl             = seg;
                buffptr          = (Buf_ctl *)ctl; 
            } else buffptr = 0;
            
            DBG(if ((getsegctl((Buf_ctl *)((char *)item - sizeof(Buf_ctl))))->allocator == NULL) abort());
            return item;
        } 

        // try to get a free item from cache
        item = getfrom_fb();
        if (item) {
            ALLSTATS(atomic_long_inc(&all_stats::instance()->hit));
            DBG(if ((getsegctl((Buf_ctl *)item))->allocator == NULL) abort());
            return ((char *)item + BUFFER_OVERHEAD);
        }
        
        ALLSTATS(atomic_long_inc(&all_stats::instance()->miss));
        
        /* if there are not available items try to allocate a new slab */
        if (newslab()<0) return NULL;
        goto avail;
        return NULL; // not reached
    }

    /* return an item into the slab */
    inline bool putitem(Buf_ctl * buf) {        
        DBG(assert(buf)); DBG(assert(nslabs>0));
        if (atomic_long_read(&nomoremalloc)) { 
            Seg_ctl  * seg = *(Seg_ctl **)buf;
            checkReclaim(seg);
            return false;
        }
        
        xThreadData * xtd = (xThreadData*)pthread_getspecific(fb_key);
        
        // if xtd is NULL probably register4free function has not yet been called
        if (!xtd) xtd = register4free();
        DBG(if (!xtd) abort());

#if defined(FF_BOUNDED_BUFFER)
        register SWSR_Ptr_Buffer * const b = xtd->b;        
        if (b) {
            register int cnt=0;
            do {
                if (b->push((char *)buf))  {
                    ALLSTATS(atomic_long_inc(&all_stats::instance()->freeok1));
                    return true;
                    //break;
                }
                else   {
                    if (cnt<TICKS_CNT) { // retry
                        ticks_wait(TICKS_TO_WAIT);
                        ++cnt;
                        continue;
                    } else {
                        /* fallback:  
                         * ok, there is no way to insert the item in the buffer,
                         * so we insert in the leak queue.
                         */
                        xtd->leak->push((void *)buf);
                        ALLSTATS(atomic_long_inc(&all_stats::instance()->nleak));
                        return false;
                    }                    
                }
            } while(1);
        }
        // I'm the allocator and this is my personal buffer.
#endif
        xtd->leak->push((void *)buf);
        ALLSTATS(atomic_long_inc(&all_stats::instance()->freeok2));
        return false;
    }

    inline const size_t getsize() const { return size; }
    inline const size_t getnslabs() const { return nslabs;}
    inline const size_t allocatedsize() const { 
        return (alloc?alloc->getallocated():0);
    }
    
protected:
    size_t                     size;         /* size of slab buffer */
    size_t                     nslabs;       /* number of buffers in the slab segment */

    xThreadData            **  fb;
    size_t                     fb_capacity;
    size_t                     fb_size;
    lock_t                     lock;
    //pthread_mutex_t            fb_mutex;
            
    Buf_ctl *             buffptr;
    size_t                availbuffers;

private:
    atomic_long_t            nomoremalloc;
    SegmentAllocator * const alloc;  
    ff_allocator     * const mainalloc;
    pthread_key_t            fb_key;
    svector<void *>          seglist;
};


class ff_allocator {
    friend class FFAllocator;
public:
    /* tell which slabs contains items of the given size */
    inline int getslabs(size_t size) {
        if (!size) return -1;
        --size;
        size_t e=1,sz=size;
        while(sz >>= 1) ++e;
        if (e < (size_t)POW2_MIN) return 0;
        if (e > (size_t)POW2_MAX) return -1;
        /*
          {
          for(int i=POW2_MAX-POW2_MIN+1;i<N_SLABBUFFER;++i)
          if ((size_t)buffersize[i]>=size) return i;
          }
        */
        return e-POW2_MIN;
    }
    
private:
    inline Seg_ctl * const getsegctl(Buf_ctl * buf) { return *((Seg_ctl **)buf); }

    inline SlabCache * const getslabs(Buf_ctl * buf) {
        return (getsegctl(buf))->cacheentry;
    }

protected:

    virtual inline void   free(Buf_ctl * buf) {
        SlabCache * const entry = getslabs(buf);        
        DBG(if (!entry) abort());
        ALLSTATS(atomic_long_inc(&all_stats::instance()->nfree));
        entry->putitem(buf);
    }

    inline size_t allocatedsize() {
        size_t s=0;
        svector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
        for(;b!=e;++b) {
            if ((*b) && (*b)->getnslabs())
                s+=(*b)->allocatedsize();
        }
        return s;
    }
    
public:
    // FIX: we have to implement max_size !!!!
    ff_allocator(size_t max_size=0):alloc(0),max_size(max_size) {}

    virtual ~ff_allocator() {
        for(size_t i=0;i<slabcache.size(); ++i) {
            if (slabcache[i]) { 
                slabcache[i]->~SlabCache();
                ::free(slabcache[i]);
                slabcache[i] = NULL;
            }
        }
        if (alloc) { 
            alloc->~SegmentAllocator();
            ::free(alloc); 
            alloc=NULL;
        }
    }
    
    // initialize the allocator
    int init(const int _nslabs[N_SLABBUFFER]=0, bool prealloc=true) {
        svector<int> nslabs; 
        if (_nslabs) 
            for (int i=0;i<N_SLABBUFFER; ++i) 
                nslabs.push_back(_nslabs[i]);
        else
            for (int i=0;i<N_SLABBUFFER; ++i) 
                nslabs.push_back(nslabs_default[i]);
        
        alloc = (SegmentAllocator *)::malloc(sizeof(SegmentAllocator));
        if (!alloc) return -1;
        new (alloc) SegmentAllocator();
        
        SlabCache * s = 0;
        slabcache.reserve(N_SLABBUFFER);
        for(int i=0; i<N_SLABBUFFER; ++i) {
            s = (SlabCache*)::malloc(sizeof(SlabCache));
            if (!s) return -1;
            new (s) SlabCache(this, alloc, buffersize[i], nslabs[i]);
            if (s->init(prealloc)<0) {               
                error("ff_allocator:init: slab init fails!\n");
                return -1;
            }
            slabcache.push_back(s);
        }
        return 0;
    }

    /* thread that allocates memory have to call this method */
    inline int registerAllocator() {
        svector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
        for(;b!=e;++b) {
            if ((*b)->getnslabs())
                if ((*b)->register4free(true)<0) return -1;
        }
        return 0;
    }

    inline void deregisterAllocator(bool reclaimMemory=true) {
        svector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
        for(;b!=e;++b) {
            if ((*b)->getnslabs())
                (*b)->deregisterAllocator(reclaimMemory);
        }
    }
    

    /* only threads different from the allocator one have to register themself to perform frees */
    inline int register4free() {
        svector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
        for(;b!=e;++b) {
            if ((*b)->getnslabs())
                if ((*b)->register4free()<0) return -1;
        }
        
        return 0; 
    }
        
    inline void * malloc(size_t size) { 
        /* use standard allocator if the size is too big or 
           we don't want to use the ff_allocator for that size
        */
        if (size>MAX_SLABBUFFER_SIZE || 
            (slabcache[getslabs(size)]->getnslabs()==0)) {
            ALLSTATS(atomic_long_inc(&all_stats::instance()->sysmalloc));
            void * ptr = ::malloc(size+sizeof(Buf_ctl));
            if (!ptr) return 0;
            ((Buf_ctl *)ptr)->ptr = 0;
            return (char *)ptr + sizeof(Buf_ctl);
        }
        int entry = getslabs(size);
        DBG(if (entry<0) abort());
        ALLSTATS(atomic_long_inc(&all_stats::instance()->nmalloc));
        void * buf = slabcache[entry]->getitem();
        return buf;
    }

    inline int posix_memalign(void **memptr, size_t alignment, size_t size) {
        if (!isPowerOfTwoMultiple(alignment, sizeof(void *))) return -1;
        
        size_t realsize = size+alignment;

        if (realsize > MAX_SLABBUFFER_SIZE ||
            (slabcache[getslabs(realsize)]->getnslabs()==0)) {
             ALLSTATS(atomic_long_inc(&all_stats::instance()->sysmalloc));
            void * ptr = ::malloc(realsize+sizeof(Buf_ctl));          
            if (!ptr) return -1;
            ((Buf_ctl *)ptr)->ptr = 0;
            void * ptraligned = (void *)(((long)((char*)ptr+sizeof(Buf_ctl)) + alignment) & ~(alignment-1));
            ((Buf_ctl *)((char *)ptraligned - sizeof(Buf_ctl)))->ptr = 0;
            *memptr = ptraligned;
            return 0;
        }
        int entry = getslabs(realsize);
        DBG(if (entry<0) abort());
        ALLSTATS(atomic_long_inc(&all_stats::instance()->nmalloc));
        void * buf = slabcache[entry]->getitem();
        Buf_ctl * backptr = (Buf_ctl *)((char *)buf - sizeof(Buf_ctl));
        DBG(assert(backptr)); 
        void * ptraligned = (void *)(((long)((char*)buf) + alignment) & ~(alignment-1));
        for(Buf_ctl *p=(Buf_ctl*)buf;p!=(Buf_ctl*)ptraligned;p++)
            p->ptr = backptr->ptr;


        *memptr = ptraligned;
        return 0;
    }

    inline void   free(void * ptr) {
        if (!ptr) return;
        Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));

        if (!buf->ptr) { 
            ALLSTATS(atomic_long_inc(&all_stats::instance()->sysfree));
            ::free(buf); 
            return; 
        }
        Buf_ctl  * buf2 = (Buf_ctl *)((char *)buf - sizeof(Buf_ctl));
        while(buf->ptr == buf2->ptr) {
            buf = buf2;
            buf2 = (Buf_ctl *)((char *)buf - sizeof(Buf_ctl));
        }
        free(buf);
    }

    inline void * realloc(void * ptr, size_t newsize) {
        if (ptr) {
            size_t oldsize;
            Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));
            if (!buf->ptr) {
                ALLSTATS(atomic_long_inc(&all_stats::instance()->sysrealloc));
                void * newptr= ::realloc(buf, newsize+sizeof(Buf_ctl));
                if (!ptr) return 0;
                ((Buf_ctl *)newptr)->ptr = 0;
                return (char *)newptr + sizeof(Buf_ctl);
            }
            ALLSTATS(atomic_long_inc(&all_stats::instance()->nrealloc));
            SlabCache * const entry = getslabs(buf);

            if (!entry) return 0;
            if ((oldsize=entry->getsize()) >= newsize) {
                return ptr;
            }
            void * newptr = this->malloc(newsize);
            memcpy(newptr,ptr,oldsize);
            entry->putitem(buf);
            return newptr;
        }
        return this->malloc(newsize);
    }
    
    /* like realloc but it doesn't copy data */
    inline void * growsup(void * ptr, size_t newsize) {
        if (ptr) {
            size_t oldsize;
            Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));
            
            if (!buf->ptr) { /* WARNING: this is probably an error !!! */
                ALLSTATS(atomic_long_inc(&all_stats::instance()->sysrealloc));
                void * newptr= ::realloc(buf, newsize+sizeof(Buf_ctl));
                if (!ptr) return 0;
                ((Buf_ctl *)newptr)->ptr = 0;
                return (char *)newptr + sizeof(Buf_ctl);
            }
            SlabCache * const entry = getslabs(buf);
            if (!entry) return 0;
            if ((oldsize=entry->getsize()) >= newsize) {
                return ptr;
            }
            entry->putitem(buf);
        }
        return this->malloc(newsize);
    }

    ALLSTATS(void printstats(std::ostream & out) { all_stats::instance()->print(out);})

private:
    svector<SlabCache *>   slabcache;
    SegmentAllocator     * alloc;
    size_t                 max_size;
};

/* ----------------------------------- */

// forward decl
class ffa_wrapper;
static void FFAkeyDestructorHandler(void * kv);
static void killMyself(ffa_wrapper * ptr);

class ffa_wrapper: public ff_allocator {
protected:

    virtual inline void   free(Buf_ctl * buf) {

        ff_allocator::free(buf);

#if 0
        std::cerr << pthread_self() << " nomorealloc= " 
                  << atomic_long_read(&nomorealloc) 
                  << " allocated size= " << allocatedsize() << "\n"; 
#endif

        // Do I have to kill myself?
        if (atomic_long_read(&nomorealloc) && 
            (allocatedsize()==0)) {
            killMyself(this);
        }
    }

public:
    friend void FFAkeyDestructorHandler(void * kv);

    // FIX: we have to implement max_size !!!!
    ffa_wrapper(size_t max_size=0):ff_allocator(max_size) {
        atomic_long_set(&nomorealloc,0);
    }
    
    virtual ~ffa_wrapper() {}

    inline void * malloc(size_t size) { 
        return ff_allocator::malloc(size);
    }

    inline int posix_memalign(void **memptr, size_t alignment, size_t size) {
        return ff_allocator::posix_memalign(memptr,alignment,size);
    }

    // just to be safe.... 
    inline void   free(void * ptr) {  abort();  }

    inline void * realloc(void * ptr, size_t newsize) {
        return ff_allocator::realloc(ptr,newsize);
    }
    
    /* like realloc but it doesn't copy data */
    inline void * growsup(void * ptr, size_t newsize) {
        return ff_allocator::growsup(ptr,newsize);
    }
private:
    inline void nomoremalloc() {
        atomic_long_set(&nomorealloc,1);
    }

private:
    atomic_long_t    nomorealloc;  
};

struct FFAxThreadData {
    FFAxThreadData(ffa_wrapper * f):f(f) {}
    ffa_wrapper * f;
    long padding[longxCacheLine-sizeof(ffa_wrapper*)];
};


/*
 * The FFAllocator.
 *
 */
class FFAllocator {
    enum {MIN_A_CAPACITY=32};
private:

    FFAllocator():AA(0),A(0),A_capacity(0),A_size(0)  {    
        init_unlocked(lock);
        if (pthread_key_create(&A_key,FFAkeyDestructorHandler)!=0) {
            error("FFAllocator FATAL ERROR: pthread_key_create fails\n");
            abort();
        }
    }
    
    ~FFAllocator() {
#if 0
        if (A) {
            for(unsigned i=0;i<A_size;++i)
                if (A[i]) deleteAllocator(A[i]);            
            ::free(A);
        }
        pthread_key_delete(A_key);
#endif
    }

    inline Seg_ctl * const getsegctl(Buf_ctl * buf) { return *((Seg_ctl **)buf); }


    inline FFAxThreadData * newAllocator(bool prealloc,
                                         int _nslabs[N_SLABBUFFER],
                                         size_t max_size) {
        FFAxThreadData * ffaxtd = (FFAxThreadData *)::malloc(sizeof(FFAxThreadData)+sizeof(ffa_wrapper));
        if (!ffaxtd) return NULL;
        ffa_wrapper * f = (ffa_wrapper*)((char *)ffaxtd+sizeof(FFAxThreadData));

        new (f) ffa_wrapper(max_size);
        new (ffaxtd) FFAxThreadData(f);

        if (f->init(_nslabs, prealloc)<0) {
            error("FFAllocator:newAllocator: init fails!\n");
            return NULL;
        }
        if (f->registerAllocator()<0) {
            error("FFAllocator:newAllocator: registerAllocator fails!\n");
            return NULL;
        }

        spin_lock(lock);
        if (A_size == A_capacity) {
            FFAxThreadData ** A_new = (FFAxThreadData**)::realloc(A,(A_capacity+MIN_A_CAPACITY)*sizeof(FFAxThreadData*));
            if (!A_new) { spin_unlock(lock); return NULL;}
            A=A_new;
            A_capacity += MIN_A_CAPACITY;
        }
        A[A_size++] = ffaxtd;
        spin_unlock(lock);

        ALLSTATS(atomic_long_inc(&all_stats::instance()->newallocator));

        return ffaxtd;
    }


public:
    
    /* return the FFAallocator object */
    static inline FFAllocator * instance() {
        static FFAllocator FFA;
        return &FFA;
    }

    /* Allocator factory method.
     * Each time this method is called you obtain a new memory allocator.
     * The calling thread is registered as an allocator thread.
     */
    inline ffa_wrapper * const newAllocator(size_t max_size=0, 
                                            int _nslabs[N_SLABBUFFER]=0,
                                            bool prealloc=true) {
        FFAxThreadData * ffaxtd = newAllocator(prealloc, _nslabs, max_size);
        if (!ffaxtd) return NULL;
        return ffaxtd->f;
    }

    inline void deleteAllocator(ffa_wrapper * f, bool reclaimMemory=true) {
        if (!f) return;

        spin_lock(lock);
        for (unsigned int i=0;i<A_size;++i) {
            if (A[i]->f == f) {
                f->deregisterAllocator(reclaimMemory);
                f->~ffa_wrapper();
                A[i]->~FFAxThreadData();
                ::free(A[i]);

                for (unsigned int j=i+1;j<A_size;++i,++j) 
                    A[i]=A[j];
                --A_size;

                break;
            }
        }
        spin_unlock(lock);
        ALLSTATS(atomic_long_inc(&all_stats::instance()->deleteallocator));
    }

    inline void deleteAllocator() {
        FFAxThreadData * ffaxtd = (FFAxThreadData*)pthread_getspecific(A_key);
        if (!ffaxtd) return;        
        deleteAllocator(ffaxtd->f,true);
    }

    
    inline void * malloc(size_t size) {
        /* use standard allocator if the size is too big */
        if (size>MAX_SLABBUFFER_SIZE) {
            ALLSTATS(atomic_long_inc(&all_stats::instance()->sysmalloc));
            void * ptr = ::malloc(size+sizeof(Buf_ctl));
            if (!ptr) return 0;
            ((Buf_ctl *)ptr)->ptr = 0;
            return (char *)ptr + sizeof(Buf_ctl);
        }

        FFAxThreadData * ffaxtd = (FFAxThreadData*)pthread_getspecific(A_key);
        if (!ffaxtd) {
            ffaxtd = newAllocator(false,0,0);

            if (!ffaxtd) {
                error("FFAllocator:malloc: newAllocator fails!\n");
                return NULL;
            }

            spin_lock(lock);
            if (pthread_setspecific(A_key, ffaxtd)!=0) {
                deleteAllocator(ffaxtd->f);
                spin_unlock(lock);
                return NULL;
            }
            spin_unlock(lock);
        }

        return ffaxtd->f->malloc(size);
    }

    inline int posix_memalign(void **memptr, size_t alignment, size_t size) {
        if (!isPowerOfTwoMultiple(alignment, sizeof(void *))) return -1;
        
        size_t realsize = size+alignment;

        if (realsize > MAX_SLABBUFFER_SIZE) {
            ALLSTATS(atomic_long_inc(&all_stats::instance()->sysmalloc));
            void * ptr = ::malloc(realsize+sizeof(Buf_ctl));          
            if (!ptr) return -1;
            ((Buf_ctl *)ptr)->ptr = 0;
            void * ptraligned = (void *)(((long)((char*)ptr+sizeof(Buf_ctl)) + alignment) & ~(alignment-1));
            ((Buf_ctl *)((char *)ptraligned - sizeof(Buf_ctl)))->ptr = 0;
            *memptr = ptraligned;
            return 0;
        }
        FFAxThreadData * ffaxtd = (FFAxThreadData*)pthread_getspecific(A_key);
        if (!ffaxtd) {
            ffaxtd = newAllocator(false,0,0);

            if (!ffaxtd) {
                error("FFAllocator:posix_memalign: newAllocator fails!\n");
                return -1;
            }

            spin_lock(lock);
            if (pthread_setspecific(A_key, ffaxtd)!=0) {
                deleteAllocator(ffaxtd->f);
                spin_unlock(lock);
                return -1;
            }
            spin_unlock(lock);
        }
        return ffaxtd->f->posix_memalign(memptr,alignment,size);
    }

    inline void   free(void * ptr) {
        if (!ptr) return;
        Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));        
        if (!buf->ptr) { 
            ALLSTATS(atomic_long_inc(&all_stats::instance()->sysfree));
            ::free(buf); 
            return; 
        }
        Buf_ctl  * buf2 = (Buf_ctl *)((char *)buf - sizeof(Buf_ctl));
        while(buf->ptr == buf2->ptr) {
            buf = buf2;
            buf2 = (Buf_ctl *)((char *)buf - sizeof(Buf_ctl));
        }
        DBG(if (!getsegctl(buf)->allocator) abort());
        (getsegctl(buf)->allocator)->free(buf);
    }

    inline void * realloc(void * ptr, size_t newsize) {
        if (ptr) {
           Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));
           if (!buf->ptr) { /* use standard allocator */
               ALLSTATS(atomic_long_inc(&all_stats::instance()->sysrealloc));
               void * newptr= ::realloc(buf, newsize+sizeof(Buf_ctl));
               if (!ptr) return 0;
               ((Buf_ctl *)newptr)->ptr = 0;
               return (char *)newptr + sizeof(Buf_ctl);
           }

           ffa_wrapper * f = (ffa_wrapper*)pthread_getspecific(A_key);
           if (!f) return NULL;
           return f->realloc(ptr,newsize);
        }
        return this->malloc(newsize);
    }
    
    /* like realloc but it doesn't copy data */
    inline void * growsup(void * ptr, size_t newsize) {
        if (ptr) {
            Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));
            if (!buf->ptr) { /* use standard allocator */
                ALLSTATS(atomic_long_inc(&all_stats::instance()->sysrealloc));
                void * newptr= ::realloc(buf, newsize+sizeof(Buf_ctl));
                if (!ptr) return 0;
                ((Buf_ctl *)newptr)->ptr = 0;
                return (char *)newptr + sizeof(Buf_ctl);
            }
            
            ffa_wrapper * f = (ffa_wrapper*)pthread_getspecific(A_key);
            if (!f) return NULL;
            return f->growsup(ptr,newsize);
        }
        return this->malloc(newsize);
    }

    ALLSTATS(void printstats(std::ostream & out) { all_stats::instance()->print(out);})

private:
    size_t          AA;
    FFAxThreadData **A;
    size_t          A_capacity;
    size_t          A_size;

    lock_t          lock;
    pthread_key_t   A_key;
};


static void FFAkeyDestructorHandler(void * kv) {
    FFAxThreadData * ffaxtd = (FFAxThreadData*)kv;
    ffaxtd->f->nomoremalloc();
    ffaxtd->f->deregisterAllocator();
}
static inline void killMyself(ffa_wrapper * ptr) {
    FFAllocator::instance()->deleteAllocator(ptr);
}


} // namespace ff

#endif /* _SLAB_ALLOCATOR_HPP_ */
