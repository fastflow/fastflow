/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef __uSWSR_PTR_BUFFER_HPP_
#define __uSWSR_PTR_BUFFER_HPP_
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License version 3 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *
 ****************************************************************************
 */
/* Single-Writer Single-Reader unbounded buffer.
 * No lock is needed around pop and push methods.
 */

#include <ff/dynqueue.hpp>
#include <ff/buffer.hpp>
#include <ff/spin-lock.hpp>
#if defined(HAVE_ATOMIC_H)
#include <asm/atomic.h>
#else
#include <ff/atomic/atomic.h>
#endif

namespace ff {

class BufferPool {
public:
    BufferPool(int cachesize)
        :inuse(cachesize),bufcache(cachesize) {
        bufcache.init();
    }
    
    ~BufferPool() {
        union { SWSR_Ptr_Buffer * b1; void * b2;} p;

        while(inuse.pop(&p.b2)) {
            p.b1->~SWSR_Ptr_Buffer();
            free(p.b2);
        }
        while(bufcache.pop(&p.b2)) {
            p.b1->~SWSR_Ptr_Buffer();	    
            free(p.b2);
        }
    }
    
    inline SWSR_Ptr_Buffer * const next_w(size_t size)  { 
        union { SWSR_Ptr_Buffer * buf; void * buf2;} p;
        if (!bufcache.pop(&p.buf2)) {
            p.buf = (SWSR_Ptr_Buffer*)malloc(sizeof(SWSR_Ptr_Buffer));
            new (p.buf) SWSR_Ptr_Buffer(size);
#if defined(uSWSR_MULTIPUSH)        
            if (p.buf->init(true)<0) return NULL;
#else
            if (p.buf->init()<0) return NULL;
#endif
        }
        
        inuse.push(p.buf);
        return p.buf;
    }
    
    inline SWSR_Ptr_Buffer * const next_r()  { 
        union { SWSR_Ptr_Buffer * buf; void * buf2;} p;
        return (inuse.pop(&p.buf2)? p.buf : NULL);
    }
    
    inline void release(SWSR_Ptr_Buffer * const buf) {
        buf->reset();
        if (!bufcache.push(buf)) {
            buf->~SWSR_Ptr_Buffer();	    
            free(buf);
        }
    }
    
private:
    dynqueue           inuse;
    SWSR_Ptr_Buffer    bufcache;
};
    

// unbounded buffer based on the SWSR_Ptr_Buffer 
class uSWSR_Ptr_Buffer {
private:
    enum {CACHE_SIZE=32};
#if defined(uSWSR_MULTIPUSH)
    enum { MULTIPUSH_BUFFER_SIZE=16};

    inline bool multipush() {
        if (buf_w->multipush(multipush_buf,MULTIPUSH_BUFFER_SIZE)) {
            mcnt=0; 
            return true;
        }

        if (fixedsize) return false;
        // try to get a new buffer             
        SWSR_Ptr_Buffer * t = pool.next_w(size);
        if (!t) return false; // EWOULDBLOCK
        buf_w = t;
        buf_w->multipush(multipush_buf,MULTIPUSH_BUFFER_SIZE);
        mcnt=0;
        return true;
    }
#endif

public:
    uSWSR_Ptr_Buffer(size_t n, const bool fixedsize=false):
        buf_r(0),buf_w(0),size(n),fixedsize(fixedsize),
        pool(CACHE_SIZE) {}
    
    ~uSWSR_Ptr_Buffer() {
        if (buf_r) {
            buf_r->~SWSR_Ptr_Buffer();
            free(buf_r);
        }
        // buf_w either is equal to buf_w or is freed by BufferPool destructor
    }
    
    bool init() {
        if (buf_w || buf_r) return false;
#if defined(uSWSR_MULTIPUSH)
        if (size<=MULTIPUSH_BUFFER_SIZE) return false;
#endif
        buf_r = (SWSR_Ptr_Buffer*)::malloc(sizeof(SWSR_Ptr_Buffer));
        if (!buf_r) return false;
        new ((void *)buf_r) SWSR_Ptr_Buffer(size);
#if defined(uSWSR_MULTIPUSH)        
        if (buf_r->init(true)<0) return false;
#else
        if (buf_r->init()<0) return false;
#endif
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
 
    /* push the input value into the queue. 
     * If fixedsize has been set, this method may
     * returns false, this means EWOULDBLOCK 
     * and the call should be retried.  
     */
    inline bool push(void * const data) {
        /* NULL values cannot be pushed in the queue */
        if (!data || !buf_w) return false;
        
        if (!available()) {

            if (fixedsize) return false;

            // try to get a new buffer             
            SWSR_Ptr_Buffer * t = pool.next_w(size);
            if (!t) return false; // EWOULDBLOCK
            buf_w = t;
        }
        //DBG(assert(buf_w->push(data)); return true;);
        buf_w->push(data);
        return true;
    }

#if defined(uSWSR_MULTIPUSH)
    /* massimot: experimental code
     *
     * This method provides the same interface of the push one but 
     * uses the multipush method to provide a batch of items to
     * the consumer thus ensuring better cache locality and 
     * lowering the cache trashing.
     */
    inline bool mpush(void * const data) {
        if (!data) return false;
        
        if (mcnt==MULTIPUSH_BUFFER_SIZE) 
            return multipush();        

        multipush_buf[mcnt++]=data;

        if (mcnt==MULTIPUSH_BUFFER_SIZE) return multipush();

        return true;
    }

    inline bool flush() {
        return (mcnt ? multipush() : true);
    }
#endif /* uSWSR_MULTIPUSH */
    
    inline bool  pop(void ** data) {
        if (!data || !buf_r) return false;
        if (buf_r->empty()) { // current buffer is empty
            if (buf_r == buf_w) return false;
            if (buf_r->empty()) { // we have to check again
                SWSR_Ptr_Buffer * tmp = pool.next_r();
                if (tmp) {
                    // there is another buffer, release the current one 
                    pool.release(buf_r); 
                    buf_r = tmp;                    
                }
            }
        }
        //DBG(assert(buf_r->pop(data)); return true;);
        return buf_r->pop(data);
    }    

private:
    // Padding is required to avoid false-sharing between 
    // core's private cache
    SWSR_Ptr_Buffer * buf_r;
    long padding1[longxCacheLine-1];
    SWSR_Ptr_Buffer * buf_w;
    long padding2[longxCacheLine-1];
#if defined(uSWSR_MULTIPUSH)
    /* massimot: experimental code (see multipush)
     *
     */
    // local multipush buffer used by the mpush method
    void  * multipush_buf[MULTIPUSH_BUFFER_SIZE];
    int     mcnt;
#endif
    const size_t      size;
    const bool        fixedsize;
    BufferPool        pool;
};
    
} // namespace


#endif /* __uSWSR_PTR_BUFFER_HPP_ */
