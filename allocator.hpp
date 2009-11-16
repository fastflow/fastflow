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


/* The ff_allocator allocates only large chunks of memory, slicing them up 
 * into little chunks all with the same size. 
 * Only one thread can perform malloc operations while any number of threads 
 * may perform frees.
 *
 * This is based on the idea of Slab Allocator, for more details about Slab 
 * Allocator please see:
 *  Bonwick, Jeff. "The Slab Allocator: An Object-Caching Kernel Memory 
 *  Allocator." Boston USENIX Proceedings, 1994.
 *
 */

#include <assert.h>
#include <vector>
#include <algorithm>
#include <deque>

#if defined(HAVE_ATOMIC_H)
#include <asm/atomic.h>
#else
#include <atomic.h>
#endif

#include <pthread.h>
#include <cycle.h>
#include <buffer.hpp>



//#define DEBUG_ALLOCATOR 1
#if defined(DEBUG_ALLOCATOR)
#define DBG(X) X
#else
#define DBG(X)
#endif

namespace ff {

enum { N_SLABBUFFER=9, MAX_SLABBUFFER_SIZE=8192};
static const int POW2_MIN  = 5;
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
    

// forward declaration
class SlabCache;

/* Each slab buffers has a Buf_ctl data structure at the beginning of the data. */
struct Buf_ctl {  SlabCache  * ptr; };

/* Each slab segment has a Seg_ctl data structure at the beginning of the data. */
struct Seg_ctl {
    DBG(size_t    refcount);      // number of buffers in use
    SlabCache *   cacheentry;     // reference to the SlabCache that owns the segment
    atomic_long_t availbuffers;   // number of items in the buffer freelist
};


/* per thread data */
struct xThreadData {
    enum { BUFFER_ITEMS=4096 };
    
    xThreadData(const bool allocator):b(0) { 
        if (!allocator) {
            b= new SWSR_Ptr_Buffer(BUFFER_ITEMS); 
            if (!b) abort();
            b->init();
        }
        leak.push_front(NULL);
    }

    ~xThreadData() { 
        if (b) delete b; 
    }
    
    SWSR_Ptr_Buffer       * b;
    std::deque<Buf_ctl *>   leak;
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
 *                          |    ----------------------------------------------------
 *                          |   |              | p |               | p | 
 *           (Slab segment) |---|- cacheentry  | t |      data     | t |      data 
 *                              |  availbuffers| r |               | r | 
 *                               ----------------------------------------------------
 *                              |    Seg_ctl   |         
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
            std::vector<void *>::iterator b(seglist.begin()), e(seglist.end());
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
                if ((buf=fb[i]->leak.front())) {
                    fb[i]->leak.pop_front();     
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
    SlabCache(SegmentAllocator * const alloc, size_t sz, std::pair<int,int> & ns):
        size(sz),nslabs(ns.first),fb(0),fb_capacity(0),fb_size(0),buffptr(0),availbuffers(0),alloc(alloc) {}
    
    ~SlabCache() { 
        std::vector<void *>::iterator b(seglist.begin()), e(seglist.end());
        for(;b!=e;++b) {
            freeslab(*b,false);
        }
        seglist.clear();
    }

    int init() {
        atomic_long_set(&nomoremalloc,0);
        pthread_mutex_init(&fb_mutex,NULL);

        fb = (xThreadData**)::malloc(MIN_FB_CAPACITY*sizeof(xThreadData*));
        if (!fb) return -1;
        fb_capacity = MIN_FB_CAPACITY;

        if (pthread_key_create(&fb_key,NULL)!=0) return -1;
        if (newslab()<0) return -1;
        return 0;
    }
    
    inline int register4free(const bool allocator=false) {
        xThreadData * xtd = (xThreadData*)pthread_getspecific(fb_key);
        if (!xtd) {
            xtd = new xThreadData(allocator);
            if (!xtd) return -1;

            pthread_mutex_lock(&fb_mutex);
            if (fb_size==fb_capacity) {
                xThreadData ** fb_new = (xThreadData**)::realloc(fb,(fb_capacity+MIN_FB_CAPACITY)*sizeof(xThreadData*));
                if (!fb_new) return -1; 
                fb = fb_new;
                fb_capacity += MIN_FB_CAPACITY;
            }
            fb[fb_size++] = xtd;
            if (pthread_setspecific(fb_key, xtd)!=0) return -1;
            pthread_mutex_unlock(&fb_mutex);

            return 0;
        }
        return 1; // already registered
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
        if (item) return ((char *)item + BUFFER_OVERHEAD);        

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

        xThreadData * const   xtd = (xThreadData*)pthread_getspecific(fb_key);

        // if it aborts here probably you don't have called 
        // the register function
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
                        xtd->leak.push_front(buf);
                        ALLSTATS(all_stats::instance()->leakinsert(xtd->leak.size()));
                        return false;
                    }                    
                }
            } while(1);
            ALLSTATS(++all_stats::instance()->freeok);
            Buf_ctl * buf_leak = xtd->leak.front();
            if (buf_leak) {
                xtd->leak.pop_front();
                bool k= b->push((char *)buf_leak);
                ALLSTATS(if (k) ++all_stats::instance()->leakremoved);
                return k;
            }
            return true;
        } else {
            xtd->leak.push_front(buf);
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
    pthread_mutex_t            fb_mutex;
            
    Buf_ctl *             buffptr;
    size_t                availbuffers;

private:
    atomic_long_t            nomoremalloc;
    SegmentAllocator * const alloc;  
    pthread_key_t            fb_key;
    std::vector<void *>      seglist;
};


class ff_allocator {
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
    inline SlabCache * const getslabs(Buf_ctl * buf) {
        Seg_ctl * seg = *((Seg_ctl **)buf);
        return seg->cacheentry;
    }

public:

    ff_allocator(size_t max_size=0):alloc(0),max_size(max_size) {}

    ~ff_allocator() {
        for(size_t i=0;i<slabcache.size(); ++i) {
            if (slabcache[i]) { 
                delete slabcache[i];
                slabcache[i] = NULL;
            }
        }
        if (alloc) { delete alloc; alloc=NULL;}
    }
    
    // initialize the allocator
    int init(std::pair<int,int> _nslabs[N_SLABBUFFER]=0) {
        std::vector<std::pair<int,int> > nslabs; 
        if (_nslabs) 
            for (int i=0;i<N_SLABBUFFER; ++i) 
                nslabs.push_back(_nslabs[i]);
        else
            for (int i=0;i<N_SLABBUFFER; ++i) 
                nslabs.push_back(std::make_pair(nslabs_default[i].nslab, nslabs_default[i].mincachedseg));
        
        alloc = new SegmentAllocator();
        if (!alloc) return -1; 
        
        SlabCache * s = 0;
        slabcache.reserve(N_SLABBUFFER);
        for(int i=0; i<N_SLABBUFFER; ++i) {
            s = new SlabCache(alloc, buffersize[i], nslabs[i]);
            if (!s) return -1; 
            if (s->init()<0) return -1;
            slabcache.push_back(s);
        }
        return 0;
    }

    inline int registerAllocator() {
        std::vector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
        for(;b!=e;++b)
            if ((*b)->register4free(true)<0) return -1;
        return 0;
    }

    inline void deregisterAllocator(bool reclaimMemory=true) {
        std::vector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
        for(;b!=e;++b)
            (*b)->deregisterAllocator(reclaimMemory);
    }
    

    /* only thread different from the allocator one have to register themself to perform free */
    inline int register4free() {
        std::vector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
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
        SlabCache * const entry = getslabs(buf);        
        DBG(if (!entry) abort());
        ALLSTATS(++all_stats::instance()->nfree);
        entry->putitem(buf);
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
    std::vector<SlabCache *>   slabcache;
    SegmentAllocator         * alloc;
    size_t                     max_size;
};

} // namespace ff

#endif /* _SLAB_ALLOCATOR_HPP_ */
