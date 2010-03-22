/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef __uSWSR_PTR_BUFFER_HPP_
#define __uSWSR_PTR_BUFFER_HPP_
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

/* Single-Writer Single-Reader unbounded buffer.
 * No lock is needed around pop and push methods.
 */

#include <buffer.hpp>
#include <squeue.hpp>
#include <spin-lock.hpp>
#if defined(HAVE_ATOMIC_H)
#include <asm/atomic.h>
#else
#include <atomic/atomic.h>
#endif

namespace ff {

class BufferPool {
public:
    BufferPool(int resizeChunk, int maxpoolsz)
        :inuse(resizeChunk),freepool(resizeChunk) {
        
        init_unlocked(freepool_lock);
        init_unlocked(inuse_lock);
        poolsize=0; maxpoolsize=maxpoolsz;
    }
    
    ~BufferPool() {
        SWSR_Ptr_Buffer * p;
        while(inuse.size()) {
            p = inuse.back();
            p->~SWSR_Ptr_Buffer();
            free(p);
            inuse.pop_back();
        }
        while(freepool.size()) {
            p = freepool.back();
            p->~SWSR_Ptr_Buffer();	    
            free(p);
            freepool.pop_back();
        }
    }
    
    inline SWSR_Ptr_Buffer * const next_w(size_t size)  { 
        SWSR_Ptr_Buffer * buf = NULL;
        spin_lock(freepool_lock);
        if (!freepool.size()) {
            spin_unlock(freepool_lock);
            
            buf = (SWSR_Ptr_Buffer*)malloc(sizeof(SWSR_Ptr_Buffer));
            new (buf) SWSR_Ptr_Buffer(size);
            if (buf->init()<0) return NULL;
            
            spin_lock(inuse_lock);
            inuse.push_back(buf);
            spin_unlock(inuse_lock);
            
            return buf;
        }
        
        buf = freepool.back();
        freepool.pop_back();
        --poolsize;
        spin_unlock(freepool_lock);
        
        spin_lock(inuse_lock);
        inuse.push_back(buf);
        spin_unlock(inuse_lock);
        
        return buf;
    }
    
    inline SWSR_Ptr_Buffer * const next_r()  { 
        SWSR_Ptr_Buffer * buf = NULL;
        spin_lock(inuse_lock);
        if (inuse.size()) {
            buf = inuse.front();
            inuse.pop_front();
        }
        spin_unlock(inuse_lock);
        
        return buf;
    }
    
    inline void release(SWSR_Ptr_Buffer * const buf) {
        if (poolsize == maxpoolsize) {
            buf->~SWSR_Ptr_Buffer();	    
            free(buf);
            return;
        }
        buf->reset();
        spin_lock(freepool_lock);
        freepool.push_back(buf);
        ++poolsize;
        spin_unlock(freepool_lock);
    }
    
    
private:
    squeue<SWSR_Ptr_Buffer *>  inuse;
    //svector<SWSR_Ptr_Buffer *> freepool;
    squeue<SWSR_Ptr_Buffer *>  freepool;
    lock_t                     inuse_lock;
    lock_t                     freepool_lock;
    int                        poolsize;
    int                        maxpoolsize;
};


// unbounded buffer based on the SWSR_Ptr_Buffer 
class uSWSR_Ptr_Buffer {

    enum {POOL_CHUNK=4096, MAX_POOL_SIZE=4};

public:
    uSWSR_Ptr_Buffer(size_t n):buf_r(0),buf_w(0),size(n),
                               pool(POOL_CHUNK,MAX_POOL_SIZE) {}
    
    ~uSWSR_Ptr_Buffer() {
        if (buf_r) {
            buf_r->~SWSR_Ptr_Buffer();
            free(buf_r);
        }
        // buf_w either is equal to buf_w or is freed by BufferPool destructor
    }
    
    
    bool init() {
        buf_r = (SWSR_Ptr_Buffer*)malloc(sizeof(SWSR_Ptr_Buffer));
        if (!buf_r) return false;
        new ((void *)buf_r) SWSR_Ptr_Buffer(size);
        if (buf_r->init()<0) return false;
        buf_w = buf_r;
        return true;
    }
    
    /* return true if the buffer is empty */
    inline bool empty() {
        return buf_r->empty();
    }
    
    /* return true if there is at least one room in the buffer */
    inline bool available()   { 
        return buf_w->available();
    }
    
    /* modify only pwrite pointer */
    inline bool push(void * const data) {
        if (!data || !buf_w) return false;
        
        if (!available()) {
            buf_w = pool.next_w(size); // getting new buffer             
            //DBG(assert(buf_w));
        }
        //DBG(assert(buf_w->push(data)); return true;);
        buf_w->push(data);
        return true;
    }
    
    inline bool  pop(void ** data) {
        if (!data || !buf_r) return false;
        if (buf_r->empty()) { // current buffer is empty
            if (buf_r == buf_w) return false;
            if (buf_r->empty()) {
                SWSR_Ptr_Buffer * tmp = pool.next_r();
                if (tmp) {
                    // there is another buffer, release the current one 
                    pool.release(buf_r); 
                    buf_r = tmp;
                    
                }
                if (buf_r->empty()) return false;
            }
        }
        //DBG(assert(buf_r->pop(data)); return true;);
        buf_r->pop(data);
        return true;
    }    
    
private:
    // Padding is required to avoid false-sharing between 
    // core's private cache
    SWSR_Ptr_Buffer * buf_r;
    long padding1[longxCacheLine-1];
    //long              padding[15];
    SWSR_Ptr_Buffer * buf_w;
    long padding2[longxCacheLine-1];
    //long              padding2[15];
    size_t            size;
    BufferPool        pool;
};
    
} // namespace


#endif /* __uSWSR_PTR_BUFFER_HPP_ */
