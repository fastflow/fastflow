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


/* Here we defined the ff_allocator and the FFAllocator. 
 * The ff_allocator allocates only large chunks of memory, slicing them up 
 * into little chunks all with the same size. Only one thread can perform malloc 
 * operations while any number of threads may perform frees. 
 * 
 * Based on ff_allocator the FFAllocator has been realised. The FFAllocator may
 * be used by any threads to dynamically allocate/deallocate memory. 
 *
 * The ff_allocator is based on the idea of Slab Allocator, for more details about Slab 
 * Allocator please see:
 *  Bonwick, Jeff. "The Slab Allocator: An Object-Caching Kernel Memory 
 *  Allocator." Boston USENIX Proceedings, 1994.
 *
 */

#include <assert.h>
#include <algorithm>

#if defined(HAVE_ATOMIC_H)
#include <asm/atomic.h>
#else
#include <atomic.h>
#endif

#include <pthread.h>
#include <cycle.h>
#include <buffer.hpp>
#include <spin-lock.hpp>
#include <svector.hpp>
#include <utils.hpp>

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
static const int buffersize[N_SLABBUFFER]={32,64,128,256,512,1024,2048,4096,8192};
static const struct { int nslab; int mincachedseg; } nslabs_default[N_SLABBUFFER] = {
    { 512, 4}, { 512, 4}, { 512, 4}, { 512, 4},
    { 128, 4}, { 64, 4}, { 32, 4}, { 16, 4}, { 8, 4}
};

#if defined(ALLOCATOR_STATS)
struct all_stats {
    unsigned long      nmalloc;
    unsigned long      nfree;
    unsigned long      freeok;
    unsigned long      nrealloc;
    unsigned long      sysmalloc;
    unsigned long      sysfree;
    unsigned long      sysrealloc;
    unsigned long      miss;
    unsigned long      freebufmiss;
    unsigned long      nleak;
    unsigned long      maxleaksize;
    unsigned long      leakremoved;
    unsigned long      memallocated;
    unsigned long      memfreed;
    unsigned long      segmalloc;
    unsigned long      segfree;


    all_stats():nmalloc(0),nfree(0),freeok(0),nrealloc(0),
                sysmalloc(0),sysfree(0),sysrealloc(0),
                miss(0),freebufmiss(0),
                nleak(0),maxleaksize(0),leakremoved(0),
                memallocated(0),memfreed(0),
                segmalloc(0),segfree(0) {}
    
    static all_stats *instance() {
        static all_stats allstats;
        return &allstats;
    }

    void leakinsert(size_t size) {
        ++nleak;
        if (size > maxleaksize) maxleaksize=size;
    }


    void print() {
        std::cout << "nmalloc     = " << nmalloc     << "\n"
                  << "miss        = " << miss        << "\n"
                  << "freebufmiss = " << freebufmiss << "\n"
                  << "nfree       = " << nfree       << "\n"
                  << "freeok      = " << freeok      << "\n"
                  << "nleak       = " << nleak       << "\n"
                  << "maxleaksize = " << maxleaksize << "\n"
                  << "leakremoved = " << leakremoved << "\n"
                  << "nrealloc    = " << nrealloc    << "\n"
                  << "memalloc    = " << memallocated<< "\n"
                  << "memfree     = " << memfreed    << "\n"
                  << "segmalloc   = " << segmalloc   << "\n"
                  << "segfree     = " << segfree     << "\n"
                  << "sysmalloc   = " << sysmalloc   << "\n"
                  << "sysfree     = " << sysfree     << "\n"
                  << "sysrealloc  = " << sysrealloc  << "\n\n";
    }
};

#define ALLSTATS(X) X

#else
#define ALLSTATS(X)
#endif


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
        void * ptr = ::malloc(segment_size);
        if (ptr) memory_allocated += segment_size;

        ALLSTATS(++all_stats::instance()->segmalloc; all_stats::instance()->memallocated+=segment_size);
        return ptr;
    }

    void   freesegment(void * ptr, size_t segment_size) { 
        DBG(assert(memory_allocated >=segment_size));
        ::free(ptr); 
        memory_allocated -= segment_size;
        ALLSTATS(++all_stats::instance()->segfree; all_stats::instance()->memfreed -= segment_size);
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
    atomic_long_t        availbuffers; 
};


/* per thread data */
struct xThreadData {
    enum { BUFFER_ITEMS=4096, LEAK_CHUNK=8192 };
    
    xThreadData(const bool allocator):b(0),leak(LEAK_CHUNK) { 
        if (!allocator) {
            b = (SWSR_Ptr_Buffer*)::malloc(sizeof(SWSR_Ptr_Buffer));
            if (!b) abort();
            new (b) SWSR_Ptr_Buffer(BUFFER_ITEMS); 
            b->init();
        }
        leak.reserve(32);
        leak.push_back(NULL);
    }

    ~xThreadData() { 
        //std::cerr << "xThreadData destructor\n";
        if (b) { b->~SWSR_Ptr_Buffer(); ::free(b);}
    }
    
    SWSR_Ptr_Buffer       * b;

    svector<Buf_ctl *>      leak;
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
    enum {TICKS_TO_WAIT=500,TICKS_CNT=3,TICKS_CNT_MAX=10,LEAK_THRESHOLD=8192};
    enum {BUFFER_OVERHEAD=sizeof(Buf_ctl)};
    enum {MIN_FB_CAPACITY=32};

    inline int newslab() {
        void * ptr = alloc->newsegment(size*nslabs + sizeof(Seg_ctl) + nslabs*BUFFER_OVERHEAD);
        
        if (ptr) {
            Seg_ctl * seg      = (Seg_ctl *)ptr;
            DBG(seg->refcount      = 0);
            seg->cacheentry    = this;
            seg->allocator     = mainalloc;
            atomic_long_set(&seg->availbuffers,0);
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
        SWSR_Ptr_Buffer * b = 0;
        for(register unsigned i=0;i<fb_size;++i) {
            DBG(assert(fb[i]));
            b = fb[i]->b;
            if (!b) {
                /* this is the xThreadData of the allocator thread */
                if ((buf=fb[i]->leak.back())) {
                    fb[i]->leak.pop_back();     
                    return buf;                // WARNING: performance vs. space 
                }                
                continue;  
            }
            if (b->pop(&buf))   {
                DBG(assert(buf));
                return buf;
            }
        }
        return 0;
    }
    
    inline void checkReclaim(Seg_ctl  * seg) {
        atomic_long_inc(&seg->availbuffers);
        if (atomic_long_read(&seg->availbuffers) == (long)nslabs) 
            freeslab(seg); /* reclaim allocated memory */
    }


    inline ticks ticks_wait(ticks t1) {
        register ticks delta, t0 = getticks();
        do { delta = getticks()-t0; } while (delta < t1);
        return delta-t1;
    }

public:    
    SlabCache(ff_allocator * const mainalloc, 
              SegmentAllocator * const alloc, size_t sz, std::pair<int,int> & ns):
        size(sz),nslabs(ns.first),fb(0),fb_capacity(0),fb_size(0),buffptr(0),availbuffers(0),
        alloc(alloc),mainalloc(mainalloc) {}
    
    ~SlabCache() { 
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

    int init() {
        atomic_long_set(&nomoremalloc,0);

        init_unlocked(lock);

        fb = (xThreadData**)::malloc(MIN_FB_CAPACITY*sizeof(xThreadData*));
        if (!fb) return -1;
        fb_capacity = MIN_FB_CAPACITY;

        if (pthread_key_create(&fb_key,NULL)!=0) return -1;

        if (newslab()<0) return -1;
        return 0;
    }
    
    inline xThreadData * register4free(const bool allocator=false) {
        xThreadData * xtd = (xThreadData*)pthread_getspecific(fb_key);
        if (!xtd) {
            xtd = (xThreadData*)::malloc(sizeof(xThreadData));
            if (!xtd) return NULL;
            new (xtd) xThreadData(allocator);

            spin_lock(lock); 
            if (fb_size==fb_capacity) {
                xThreadData ** fb_new = (xThreadData**)::realloc(fb,(fb_capacity+MIN_FB_CAPACITY)*sizeof(xThreadData*));
                if (!fb_new) return NULL; 
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
        atomic_long_set(&nomoremalloc,1);

        char * buf = 0;
        SWSR_Ptr_Buffer * b = 0;
        for(register unsigned i=0;i<fb_size;++i) {
            DBG(assert(fb[i]));
            b = fb[i]->b;
            if (b) {
                if (reclaim) 
                    while(b->pop((void **)&buf)) {
                        DBG(assert(buf));
                        Seg_ctl  * seg = *(Seg_ctl **)buf;
                        checkReclaim(seg);
                    }
                else  /* NOTE: free at least some buffer rooms to avoid race conditions for buffer full */
                    for(register int i=0;i<5;++i) b->pop((void**)&buf);
            }
        }
    }
                                               
    /* get an item from slab */
    inline void * getitem() { 
        void * item = 0;

        item = getfrom_fb();
        if (item) {
            DBG(if ((getsegctl((Buf_ctl *)item))->allocator == NULL) abort());
            return ((char *)item + BUFFER_OVERHEAD);
        }

        ALLSTATS(++all_stats::instance()->miss);

        /* if there are not available items try to allocate a new slab */
        if (!availbuffers) { 
            ALLSTATS(++all_stats::instance()->freebufmiss);
            newslab();     
        }

        /* get item from slab */
        if (availbuffers) {     
            Seg_ctl * seg = *((Seg_ctl **)buffptr);
            DBG(assert(seg));
            DBG(++seg->refcount);

            DBG(if (seg->refcount==1) assert(availbuffers==nslabs));

            item = (char *)buffptr + BUFFER_OVERHEAD;   // set data pointer            
            if (--availbuffers) {            
                Seg_ctl ** ctl   = (Seg_ctl **)((char *)item + size);
                *ctl             = seg;
                buffptr          = (Buf_ctl *)ctl; 
            } else buffptr = 0;
        } 
                    
        DBG(if ((getsegctl((Buf_ctl *)item))->allocator == NULL) abort());
        return item;
    }

    /* return an item into the slab */
    inline bool putitem(Buf_ctl * buf) {        
        DBG(assert(buf));
        if (atomic_long_read(&nomoremalloc)) { 
            Seg_ctl  * seg = *(Seg_ctl **)buf;
            checkReclaim(seg);
            return false;
        }

        xThreadData * xtd = (xThreadData*)pthread_getspecific(fb_key);

        // if xtd is NULL probably register4free function has not yet been called
        if (!xtd) xtd = register4free();
        DBG(if (!xtd) abort());

        SWSR_Ptr_Buffer * const b = xtd->b;
        
        if (b) {
            register int cnt=0;
            do {
                if (b->push((char *)buf))  break;
                else   {
                    if (cnt<TICKS_CNT) {
                        ticks_wait(TICKS_TO_WAIT);
                        ++cnt;
                    } else {
                        sched_yield();
                        if (b->push((char *)buf)) break; //return true;
                        
                        if (xtd->leak.size() >= LEAK_THRESHOLD) {
                            ++cnt;
                            if (cnt < TICKS_CNT_MAX) continue;
                        }
                        /* fallback:  insert in the leak queue */
                        xtd->leak.push_back(buf);

                        ALLSTATS(all_stats::instance()->leakinsert(xtd->leak.size()));
                        return false;
                    }                    
                }
            } while(1);
            ALLSTATS(++all_stats::instance()->freeok);
            Buf_ctl * buf_leak = xtd->leak.back();
            if (buf_leak) {
                xtd->leak.pop_back();
                bool k= b->push((char *)buf_leak);
                ALLSTATS(if (k) ++all_stats::instance()->leakremoved);
                return k;
            }
            return true;
        } else {
            xtd->leak.push_back(buf);
        }
        return false;
    }

    inline const int getsize() const { return size; }
    
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

private:
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

    inline Seg_ctl * const getsegctl(Buf_ctl * buf) { return *((Seg_ctl **)buf); }

    inline SlabCache * const getslabs(Buf_ctl * buf) {
        return (getsegctl(buf))->cacheentry;
    }

protected:

    virtual inline void   free(Buf_ctl * buf) {
        SlabCache * const entry = getslabs(buf);        
        DBG(if (!entry) abort());
        ALLSTATS(++all_stats::instance()->nfree);
        entry->putitem(buf);
    }

public:
    // FIX: we have to implement max_size !!!!
    ff_allocator(size_t max_size=0):alloc(0),max_size(max_size) {}

    ~ff_allocator() {
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
    int init(std::pair<int,int> _nslabs[N_SLABBUFFER]=0) {
        svector<std::pair<int,int> > nslabs; 
        if (_nslabs) 
            for (int i=0;i<N_SLABBUFFER; ++i) 
                nslabs.push_back(_nslabs[i]);
        else
            for (int i=0;i<N_SLABBUFFER; ++i) 
                nslabs.push_back(std::make_pair(nslabs_default[i].nslab, nslabs_default[i].mincachedseg));
        
        alloc = (SegmentAllocator *)::malloc(sizeof(SegmentAllocator));
        if (!alloc) return -1;
        new (alloc) SegmentAllocator();
        
        SlabCache * s = 0;
        slabcache.reserve(N_SLABBUFFER);
        for(int i=0; i<N_SLABBUFFER; ++i) {
            s = (SlabCache*)::malloc(sizeof(SlabCache));
            if (!s) return -1;
            new (s) SlabCache(this, alloc, buffersize[i], nslabs[i]);
            if (s->init()<0) {
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
        for(;b!=e;++b)
            if ((*b)->register4free(true)<0) return -1;
        return 0;
    }

    inline void deregisterAllocator(bool reclaimMemory=true) {
        svector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
        for(;b!=e;++b)
            (*b)->deregisterAllocator(reclaimMemory);
    }
    

    /* only threads different from the allocator one have to register themself to perform frees */
    inline int register4free() {
        svector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
        for(;b!=e;++b)
            if ((*b)->register4free()<0) return -1;
        
        return 0; 
    }
    
    inline void * malloc(size_t size) { 
        /* use standard allocator if the size is too big */
        if (size>MAX_SLABBUFFER_SIZE) {
            ALLSTATS(++all_stats::instance()->sysmalloc);
            void * ptr = ::malloc(size+sizeof(Buf_ctl));
            if (!ptr) return 0;
            ((Buf_ctl *)ptr)->ptr = 0;
            return (char *)ptr + sizeof(Buf_ctl);
        }
        int entry = getslabs(size);
        DBG(if (entry<0) abort());
        ALLSTATS(++all_stats::instance()->nmalloc);
        void * buf = slabcache[entry]->getitem();
        return buf;
    }

    inline void   free(void * ptr) {
        Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));
        
        if (!buf->ptr) { 
            ALLSTATS(++all_stats::instance()->sysfree);
            ::free(buf); 
            return; 
        }
        free(buf);
    }

    inline void * realloc(void * ptr, size_t newsize) {
        if (ptr) {
            size_t oldsize;
            Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));
            if (!buf->ptr) {
                ALLSTATS(++all_stats::instance()->sysrealloc);
                void * newptr= ::realloc(buf, newsize+sizeof(Buf_ctl));
                if (!ptr) return 0;
                ((Buf_ctl *)newptr)->ptr = 0;
                return (char *)newptr + sizeof(Buf_ctl);
            }
            ALLSTATS(++all_stats::instance()->nrealloc);
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
            
            if (!buf) return ::realloc(buf,newsize); /* WARNING: this is probably an error !!! */
            
            SlabCache * const entry = getslabs(buf);
            if (!entry) return 0;
            if ((oldsize=entry->getsize()) >= newsize) {
                return ptr;
            }
            entry->putitem(buf);
        }
        return this->malloc(newsize);
    }

    ALLSTATS(void printstats() { all_stats::instance()->print();})

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
        atomic_long_dec(&alloc_cnt);

#if 0
        std::cerr << pthread_self() << " free alloc_cnt=" << 
            atomic_long_read(&alloc_cnt) << " nomorealloc= " << 
            atomic_long_read(&nomorealloc) << "\n";
#endif

        ff_allocator::free(buf);

        if (atomic_long_read(&nomorealloc) && 
            !atomic_long_read(&alloc_cnt)) {             

            killMyself(this);
        }
    }

public:
    friend void FFAkeyDestructorHandler(void * kv);

    // FIX: we have to implement max_size !!!!
    ffa_wrapper(size_t max_size=0):ff_allocator(max_size) {
        atomic_long_set(&alloc_cnt,0);
        atomic_long_set(&nomorealloc,0);
    }
    
    inline void * malloc(size_t size) { 
        atomic_long_inc(&alloc_cnt);
        return ff_allocator::malloc(size);
    }

    inline void   free(void * ptr) {
        abort();
    }

    inline void * realloc(void * ptr, size_t newsize) {
        atomic_long_inc(&alloc_cnt);
        return ff_allocator::realloc(ptr,newsize);
    }
    
    /* like realloc but it doesn't copy data */
    inline void * growsup(void * ptr, size_t newsize) {
        atomic_long_inc(&alloc_cnt);
        return ff_allocator::growsup(ptr,newsize);
    }
private:
    inline void nomoremalloc() {
        //std::cerr << pthread_self() << " NOMOREALLOC alloc_cnt=" << 
        // atomic_long_read(&alloc_cnt) << "\n";
        atomic_long_set(&nomorealloc,1);
    }

private:
    atomic_long_t    alloc_cnt;
    atomic_long_t    nomorealloc;  
};


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
        if (A) {
            for(unsigned i=0;i<A_size;++i)
                if (A[i]) deleteAllocator(A[i]);            
            ::free(A);
        }
        pthread_key_delete(A_key);
    }

    inline Seg_ctl * const getsegctl(Buf_ctl * buf) { return *((Seg_ctl **)buf); }

public:
    
    static inline FFAllocator * instance() {
        static FFAllocator FFA;
        return &FFA;
    }

    /* Allocator factory method.
     * Each time this method is called you obtain a new memory allocator.
     * The calling thread is registered as an allocator thread.
     */
    inline ffa_wrapper * const newAllocator(size_t max_size=0, std::pair<int,int> _nslabs[N_SLABBUFFER]=0) {
        ffa_wrapper * f = (ffa_wrapper*)::malloc(sizeof(ffa_wrapper));
        if (!f) return NULL;
        new (f) ffa_wrapper(max_size);
        if (f->init(_nslabs)<0) {
            error("FFAllocator:newAllocator: init fails!\n");
            return NULL;
        }
        if (f->registerAllocator()<0) {
            error("FFAllocator:newAllocator: registerAllocator fails!\n");
            return NULL;
        }

        spin_lock(lock);
        if (A_size == A_capacity) {
            ffa_wrapper ** A_new = (ffa_wrapper**)::realloc(A,(A_capacity+MIN_A_CAPACITY)*sizeof(ffa_wrapper*));
            if (!A_new) return NULL;
            A=A_new;
            A_capacity += MIN_A_CAPACITY;
        }
        A[A_size++] = f;
        spin_unlock(lock);

        return f;
    }

    inline void deleteAllocator(ffa_wrapper * f, bool reclaimMemory=true) {
        if (!f) return;

        spin_lock(lock);
        for (unsigned int i=0;i<A_size;++i) {
            if (A[i] == f) {
                f->deregisterAllocator(reclaimMemory);
                f->~ffa_wrapper();
                ::free(f);

                for (unsigned int j=i+1;j<A_size;++i,++j) 
                    A[i]=A[j];
                --A_size;

                break;
            }
        }
        spin_unlock(lock);
    }

    inline void deleteAllocator() {
        ffa_wrapper * f = (ffa_wrapper*)pthread_getspecific(A_key);
        if (!f) return;        
        deleteAllocator(f,true);
    }

    
    inline void * malloc(size_t size) { 
        ffa_wrapper * f = (ffa_wrapper*)pthread_getspecific(A_key);
        if (!f) {
            f = newAllocator();

            if (!f) {
                error("FFAllocator:malloc: newAllocator fails!\n");
                return NULL;
            }
            spin_lock(lock);
            if (pthread_setspecific(A_key, f)!=0) {
                deleteAllocator(f);
                return NULL;
            }
            spin_unlock(lock);
        }


        return f->malloc(size);
    }

    inline void   free(void * ptr) {
        Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));        
        if (!buf->ptr) { 
            ::free(buf);  // FIX: stats!!!!
            return; 
        }
        DBG(if (!getsegctl(buf)->allocator) abort());
        (getsegctl(buf)->allocator)->free(buf);
    }

    inline void * realloc(void * ptr, size_t newsize) {
        if (ptr) {
            ffa_wrapper * f = (ffa_wrapper*)pthread_getspecific(A_key);
            if (!f) return NULL;
            return f->realloc(ptr,newsize);
        }
        return this->malloc(newsize);
    }
    
    /* like realloc but it doesn't copy data */
    inline void * growsup(void * ptr, size_t newsize) {
        if (ptr) {
            ffa_wrapper * f = (ffa_wrapper*)pthread_getspecific(A_key);
            if (!f) return NULL;
            return f->growsup(ptr,newsize);
        }
        return this->malloc(newsize);
    }

private:
    size_t          AA;
    ffa_wrapper ** A;
    size_t          A_capacity;
    size_t          A_size;

    lock_t          lock;
    pthread_key_t   A_key;
};


static void FFAkeyDestructorHandler(void * kv) {
    ffa_wrapper * ffa = (ffa_wrapper*)kv;
    ffa->nomoremalloc();
}
static inline void killMyself(ffa_wrapper * ptr) {
    FFAllocator::instance()->deleteAllocator(ptr);
}


} // namespace ff

#endif /* _SLAB_ALLOCATOR_HPP_ */
