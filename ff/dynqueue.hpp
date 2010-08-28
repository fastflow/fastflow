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

/* Dynamic Single-Writer Single-Reader unbounded queue.
 * No lock is needed around pop and push methods.
 */

#include <stdlib.h>
#include <ff/buffer.hpp>
#include <ff/sysdep.h>

namespace ff {

class dynqueue {
private:
    struct Node {
        void        * data;
        struct Node * next;
    };

    volatile Node *    head;
    long padding1[longxCacheLine-sizeof(Node *)];
    volatile Node *    tail;
    long padding2[longxCacheLine-sizeof(Node*)];
    SWSR_Ptr_Buffer    cache;
private:
    inline Node * allocnode() {
        Node * n = NULL;
        if (cache.pop((void **)&n)) return n;
        n = (Node *)::malloc(sizeof(Node));
        return n;
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
        Node * n = NULL;
        while(cache.pop((void**)&n)) free(n);
        while(head != tail) {
            n = (Node*)head;
            head = head->next;
            free(n);
        }
        if (head) free((void*)head);
    }

    inline bool push(void * const data) {
        Node * n = allocnode();
        n->data = data; n->next = NULL;
        WMB();
        tail->next = n;
        tail       = n;

        return true;
    }

    inline bool  pop(void ** data) {        
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

} // namespace

#endif /* __FF_DYNQUEUE_HPP_ */
