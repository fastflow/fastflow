/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef __FF_DYNQUEUE_HPP_ 
#define __FF_DYNQUEUE_HPP_ 
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

/* Dynamic (list-based) Single-Writer Single-Reader unbounded queue.
 * No lock is needed around pop and push methods.
 *
 * See also ubuffer.hpp for a SWSR unbounded queue.
 *
 */

#include <stdlib.h>
#include <ff/buffer.hpp>
#include <ff/sysdep.h>

namespace ff {


#if !defined(_FF_DYNQUEUE_OPTIMIZATION)
class dynqueue {
private:
    struct Node {
        void        * data;
        struct Node * next;
    };

    volatile Node *    head;
    long padding1[longxCacheLine-(sizeof(Node *)/sizeof(long))];
    volatile Node *    tail;
    long padding2[longxCacheLine-(sizeof(Node*)/sizeof(long))];
    SWSR_Ptr_Buffer    cache;
private:
    inline Node * allocnode() {
        union { Node * n; void * n2; } p;
        if (cache.pop(&p.n2)) return p.n;
        p.n = (Node *)::malloc(sizeof(Node));
        return p.n;
    }

public:
    enum {DEFAULT_CACHE_SIZE=1024};

    dynqueue(int cachesize=DEFAULT_CACHE_SIZE, bool fillcache=false):cache(cachesize) {
        Node * n = (Node *)::malloc(sizeof(Node));
        n->data = NULL; n->next = NULL;
        head=tail=n;
        cache.init();
        if (fillcache) {
            for(int i=0;i<cachesize;++i) {
                n = (Node *)::malloc(sizeof(Node));
                if (n) cache.push(n);
            }
        }
    }

    ~dynqueue() {
        union { Node * n; void * n2; } p;
        while(cache.pop(&p.n2)) free(p.n);
        while(head != tail) {
            p.n = (Node*)head;
            head = head->next;
            free(p.n);
        }
        if (head) free((void*)head);
    }

    inline bool push(void * const data) {
        if (!data) return false;
        Node * n = allocnode();
        n->data = data; n->next = NULL;
        WMB();
        tail->next = n;
        tail       = n;

        return true;
    }

    inline bool  pop(void ** data) {        
        if (!data) return false;
        if (head->next) {
            Node * n = (Node *)head;
            *data    = (head->next)->data;
            head     = head->next;

            if (!cache.push(n)) free(n);
            return true;
        }
        return false;
    }    
};

#else // _FF_DYNQUEUE_OPTIMIZATION


class dynqueue {
private:
    struct Node {
        void        * data;
        struct Node * next;
    };

    volatile Node *         head;
    volatile unsigned long  pwrite;
    long padding1[longxCacheLine-((sizeof(Node *)+sizeof(unsigned long))/sizeof(long))];
    volatile Node *        tail;
    volatile unsigned long pread;
    long padding2[longxCacheLine-((sizeof(Node*)+sizeof(unsigned long))/sizeof(long))];

    const   size_t cachesize;
    void ** cache;

private:

    inline bool cachepush(void * const data) {
        
        if (!cache[pwrite]) {
            /* Write Memory Barrier: ensure all previous memory write 
             * are visible to the other processors before any later
             * writes are executed.  This is an "expensive" memory fence
             * operation needed in all the architectures with a weak-ordering 
             * memory model where stores can be executed out-or-order 
             * (e.g. Powerpc). This is a no-op on Intel x86/x86-64 CPUs.
             */
            WMB(); 
            cache[pwrite] = data;
            pwrite += (pwrite+1 >= cachesize) ? (1-cachesize): 1;
            return true;
        }
        return false;
    }

    inline bool  cachepop(void ** data) {
        if (!cache[pread]) return false;
        
        *data = cache[pread];
        cache[pread]=NULL;
        pread += (pread+1 >= cachesize) ? (1-cachesize): 1;    
        return true;
    }    
    
public:
    enum {DEFAULT_CACHE_SIZE=1024};

    dynqueue(int cachesize=DEFAULT_CACHE_SIZE, bool fillcache=false):cachesize(cachesize) {
        Node * n = (Node *)::malloc(sizeof(Node));
        n->data = NULL; n->next = NULL;
        head=tail=n;


#if __MAC_OS_X_VERSION_MIN_REQUIRED < __MAC_10_6
        cache = (void **)::malloc(cachesize*sizeof(void*));
        if (!cache) abort();
#else
        void * ptr;
        if (posix_memalign(&ptr,longxCacheLine*sizeof(long),cachesize*sizeof(void*))!=0)
            abort();
        cache=(void **)ptr;
#endif

        if (fillcache) {
            for(int i=0;i<cachesize;++i) {
                n = (Node *)::malloc(sizeof(Node));
                if (n) cachepush(n);
            }
        }
    }

    ~dynqueue() {
        union { Node * n; void * n2; } p;
        while(cachepop(&p.n2)) free(p.n);
        while(head != tail) {
            p.n = (Node*)head;
            head = head->next;
            free(p.n);
        }
        if (head) free((void*)head);
    }

    inline bool push(void * const data) {
        if (!data) return false;

        union { Node * n; void * n2; } p;
        if (!cachepop(&p.n2))
            p.n = (Node *)::malloc(sizeof(Node));
        
        p.n->data = data; p.n->next = NULL;
        WMB();
        tail->next = p.n;
        tail       = p.n;

        return true;
    }

    inline bool  pop(void ** data) {
        if (!data) return false;
        if (head->next) {
            Node * n = (Node *)head;
            *data    = (head->next)->data;
            head     = head->next;

            if (!cachepush(n)) free(n);
            return true;
        }
        return false;
    }    
};

#endif // _FF_DYNQUEUE_OPTIMIZATION

} // namespace

#endif /* __FF_DYNQUEUE_HPP_ */
