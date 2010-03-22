/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef __SWSR_PTR_BUFFER_HPP_
#define __SWSR_PTR_BUFFER_HPP_
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

/* Single-Writer Single-Reader circular buffer.
 * No lock is needed around pop and push methods.
 * 
 * A NULL value is used to indicate buffer full and 
 * buffer empty conditions.
 *
 */

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <sysdep.h>

namespace ff {
// 64bytes is the common size of a cache line
static const int longxCacheLine = (64/sizeof(long));

class SWSR_Ptr_Buffer {
private:
    // Padding is required to avoid false-sharing between 
    // core's private cache
    volatile unsigned long    pread;
    long padding1[longxCacheLine-1];
    volatile unsigned long    pwrite;
    long padding2[longxCacheLine-1];
    const    size_t           size;
    void                   ** buf;
    
public:
    SWSR_Ptr_Buffer(size_t n):
        pread(0),pwrite(0),size(n),buf(0) {
    }
    
    ~SWSR_Ptr_Buffer() { if (buf)::free(buf); }
    
    /* initialize the circular buffer. */
    bool init() {
        if (buf) return false;

        buf = (void **)::malloc(size*sizeof(void*));
        if (!buf) return false;
        bzero(buf,size*sizeof(void*));

        return true;
    }

    /* return true if the buffer is empty */
    inline bool empty() {
        return (buf[pread]==NULL);
    }
    
    /* return true if there is at least one room in the buffer */
    inline bool available()   { 
        return (buf[pwrite]==NULL);
    }

    inline size_t buffersize() const { return size; };
    
    /* modify only pwrite pointer */
    inline bool push(void * const data) {
        if (!data) return false;
        
        if (available()) {
            /* Write Memory Barrier: ensure all previous memory write 
             * are visible to the other processors before any later
             * writes are executed.  This is an "expensive" memory fence
             * operation needed in all the architectures with a weak-ordering 
             * memory model where stores can be executed out-or-order 
             * (e.g. Powerpc). This is a no-op on Intel x86/x86-64 CPUs.
             */
            WMB(); 
            buf[pwrite] = data;
            pwrite += (pwrite+1 >= size) ? (1-size): 1;
            return true;
        }
        return false;
    }

    /* like pop but doesn't copy any data */
    inline bool  inc() {
        buf[pread]=NULL;
        pread += (pread+1 >= size) ? (1-size): 1;        
        return true;
    }    
    
    /* modify only pread pointer */
    inline bool  pop(void ** data) {
        if (!data || empty()) return false;
        
        *data = buf[pread];
        return inc();
    }    
    
    inline void * const top() const { 
        return buf[pread];  
    }    

    inline void reset() { pread=pwrite=0; bzero(buf,size*sizeof(void*));}
};

} // namespace

#endif /* __SWSR_PTR_BUFFER_HPP_ */
