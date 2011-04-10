/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef __FF_MPMCQUEUE_HPP_ 
#define __FF_MPMCQUEUE_HPP_ 

/* 
 * This file contains Multi-Producer/Multi-Consumer queue implementations.
 *
 *
 *  -- Massimo Torquati <torquati@di.unipi.it> <massimotor@gmail.com>
 *
 */


#include <stdlib.h>
#include <ff/buffer.hpp>
#include <ff/sysdep.h>
#include <ff/allocator.hpp>
#include <ff/atomic/abstraction_dcas.h>

#if defined(HAVE_CDSLIB)
#include <cds/gc/hzp/gc.h>
#include <cds/queue/msqueue_hzp.h>
#endif

namespace ff {

#define CAS abstraction_cas

/* 
 * MSqueue is an implementation of the lock-free FIFO MPMC queue by
 * Maged M. Michael and Michael L. Scott described in the paper:
 *   "Simple, Fast, and Practical Non-Blocking and Blocking
 *    Concurrent Queue Algorithms", PODC 1996.
 *
 * The MSqueue implementation is inspired to the one in the liblfds 
 * libraly that is a portable, license-free, lock-free data structure 
 * library written in C. The liblfds implementation uses double-word CAS 
 * (aka DCAS) whereas this implementation uses only single-word CAS 
 * since it relies on a implementation of a memory allocator (used to 
 * allocate internal queue nodes) which implements a deferred reclamation 
 * algorithm able to solve both the ABA problem and the dangling pointer 
 * problem.
 *
 * More info about liblfds can be found at http://www.liblfds.org
 *
 */
class MSqueue {
private:
    enum {MSQUEUE_PTR=0 };

    // forward decl of Node type
    struct Node;
 
    struct Pointer {
        Pointer() { ptr[MSQUEUE_PTR]=0;}

        inline bool operator !() {
            return (ptr[MSQUEUE_PTR]==0);
        }
        inline Pointer& operator=(const Pointer & p) {
            ptr[MSQUEUE_PTR]=p.ptr[MSQUEUE_PTR];
            return *this;
        }

        inline Pointer& operator=(Node & node) {
            ptr[MSQUEUE_PTR]=&node;
            return *this;
        }

        inline Pointer & getNodeNext() {
            return ptr[MSQUEUE_PTR]->next;
        }
        inline Node * getNode() { return  ptr[MSQUEUE_PTR]; }

        inline bool operator==( const Pointer& r ) const {
            return ((ptr[MSQUEUE_PTR]==r.ptr[MSQUEUE_PTR]));
        }

        inline operator volatile atom_t * () const { 
            union { Node* const volatile* p1; volatile atom_t * p2;} pn;
            pn.p1 = ptr;
            return pn.p2; 
        }
        inline operator atom_t * () const { 
            union { Node* const volatile* p1; atom_t * p2;} pn;
            pn.p1 = ptr;
            return pn.p2; 
        }
        
        inline operator atom_t () const { 
            union { Node* volatile p1; atom_t p2;} pn;
            pn.p1 = ptr[MSQUEUE_PTR];
            return pn.p2; 
        }

        inline void set(Node & node) {
            ptr[MSQUEUE_PTR]=&node;
        }

        inline void * getData() const { return ptr[MSQUEUE_PTR]->getData(); }

        Node * volatile ptr[1];
    } ALIGN_TO(ALIGN_SINGLE_POINTER);

    struct Node {
        Node():data(0) { next.ptr[MSQUEUE_PTR]=0;}
        Node(void * data):data(data) {
            next.ptr[MSQUEUE_PTR]=0;
        }
        
        inline operator atom_t * () const { return (atom_t *)next; }

        inline void   setData(void * const d) { data=d;}
        inline void * getData() const { return data; }

        Pointer   next;
        void    * data;
    } ALIGN_TO(ALIGN_DOUBLE_POINTER);

    Pointer  head;
    long padding1[longxCacheLine-(sizeof(Pointer)/sizeof(long))];
    Pointer  tail;
    long padding2[longxCacheLine-(sizeof(Pointer)/sizeof(long))];
    FFAllocator *delayedAllocator;

private:
    inline void allocnode(Pointer & p, void * data) {
        union { Node * p1; void * p2;} pn;

        if (delayedAllocator->posix_memalign((void**)&pn.p2,ALIGN_DOUBLE_POINTER,sizeof(Node))!=0) {
            abort();
        }            
        new (pn.p2) Node(data);
        p.set(*pn.p1);
    }

    inline void deallocnode( Node * n) {
        n->~Node();
        delayedAllocator->free(n);
    }

public:
    MSqueue(): delayedAllocator(NULL) { }
    
    ~MSqueue() {
        if (delayedAllocator)  {
            delete delayedAllocator;
            delayedAllocator = NULL;
        }
    }

    MSqueue& operator=(const MSqueue& v) { 
        head=v.head;
        tail=v.tail;
        return *this;
    }

    /* initialize the MSqueue */
    int init() {
        if (delayedAllocator) return 0;
        delayedAllocator = new FFAllocator(2); 
        if (!delayedAllocator) {
            std::cerr << "ERROR: MSqueue, cannot allocate FFAllocator!!!\n";
            return -1;
        }

        // create the first NULL node 
        // so the queue is never really empty
        Pointer dummy;
        allocnode(dummy,NULL);
        
        head = dummy;
        tail = dummy;
        return 1;
    }

    // insert method, it never fails
    inline bool push(void * const data) {
        bool done = false;

        ALIGN_TO(ALIGN_SINGLE_POINTER) Pointer tailptr;
        ALIGN_TO(ALIGN_SINGLE_POINTER) Pointer next;
        ALIGN_TO(ALIGN_SINGLE_POINTER) Pointer node;
        allocnode(node,data);

        do {
            tailptr = tail;

            if (tailptr == tail) {
                next    = tailptr.getNodeNext();
                if (!next) { // tail was pointing to the last node
                    done = (CAS((volatile atom_t *)(tailptr.getNodeNext()), 
                                (atom_t)node, 
                                (atom_t)next) == (atom_t)next);
                } else {     // tail was not pointing to the last node
                    CAS((volatile atom_t *)tail, (atom_t)next, (atom_t)tailptr);
                }
            }
        } while(!done);
        CAS((volatile atom_t *)tail, (atom_t)node, (atom_t) tailptr);
        return true;
    }
    
    // extract method, it returns false if the queue is empty
    inline bool  pop(void *& data) {        
        bool done = false;

        ALIGN_TO(ALIGN_SINGLE_POINTER) Pointer headptr;
        ALIGN_TO(ALIGN_SINGLE_POINTER) Pointer tailptr;
        ALIGN_TO(ALIGN_SINGLE_POINTER) Pointer next;

        do {
            headptr = head;
            tailptr = tail;

            if (head == headptr) {
                next    = headptr.getNodeNext();
                if (headptr.getNode() == tailptr.getNode()) {
                    if (!next) return false; // empty
                    CAS((volatile atom_t *)tail, (atom_t)next, (atom_t)tailptr);
                } else {
                    data = next.getData();
                    done = (CAS((volatile atom_t *)head, (atom_t)next, (atom_t)headptr) == (atom_t)headptr);
                }
            }
        } while(!done);

        deallocnode(headptr.getNode());
        return true;
    } 

    // return true if the queue is empty 
    inline bool empty() { 
        if ((head.getNode() == tail.getNode()) && !(head.getNodeNext()))
            return true;
        return false;            
    }
};



/*
 * Simple and scalable Multi-Producer/Multi-Consumer queue.
 * By defining at compile time MULTI_MPMC_RELAX_FIFO_ORDERING it is possible 
 * to improve performance relaxing FIFO ordering in the pop method.
 *
 * The underling MPMC queue (the Q template parameter) should export at least 
 * the following methods:
 *
 *   bool push(T)
 *   bool pop(T&)
 *   bool empty() 
 *
 *
 */
template <typename Q>
class scalableMPMCqueue {
public:
    enum {DEFAULT_POOL_SIZE=4};

    scalableMPMCqueue():stop(true) {
        atomic_long_set(&enqueue,0);
        atomic_long_set(&dequeue,0);
    }
    
    int init(size_t poolsize = DEFAULT_POOL_SIZE) {
        if (poolsize > pool.size()) {
            pool.resize(poolsize);
        }
        
        // WARNING: depending on Q, pool elements may need to be initialized  

        stop = false;
        return 1;
    }

    inline void stop_producing() { 
        WMB();
        stop=true;
    }

    // insert method, it never fails if data is not NULL
    inline bool push(void * const data) {
        if (stop) return false;
        register long q = atomic_long_inc_return(&enqueue) % pool.size();
        bool r = pool[q].push(data);
        return r;
    }

    // extract method, it returns false if the queue is empty
    inline bool  pop(void *& data) {      
        bool r;

#if !defined(MULTI_MPMC_RELAX_FIFO_ORDERING)
        //
        // enforce FIFO ordering for the consumers
        //
        register long q, q1;
        do {
            q  = atomic_long_read(&dequeue), q1 = atomic_long_read(&enqueue);            
            if ((q == q1) && !stop)  return false;
            if (CAS((volatile atom_t *)&dequeue, (atom_t)(q+1), (atom_t)q) == (atom_t)q) break;
        } while(1);

        q %= pool.size(); 
        do ; while( !(r=pool[q].pop(data)) && !stop);
        
#else  // MULTI_MPMC_RELAX_FIFO_ORDERING
        register long q = atomic_long_inc_return(&dequeue) % pool.size();
        r = pool[q].pop(data);
#endif
        
        if (stop && !r && empty())  { 
            data = (void*)FF_EOS;   
            return true; 
        }
        
        return r;
    }
    
    inline bool empty() {
        for(size_t i=0;i<pool.size();++i)
            if (!pool[i].empty()) return false;
         return true;
    }
private:
    atomic_long_t enqueue;
    long padding1[longxCacheLine-sizeof(atomic_long_t)];
    atomic_long_t dequeue;
    long padding2[longxCacheLine-sizeof(atomic_long_t)];
protected:
    std::vector<Q> pool;
    bool          stop;
};

/* 
 * multiMSqueue is a specialization of the scalableMPMCqueue which uses the MSqueue 
*/
class multiMSqueue: public scalableMPMCqueue<MSqueue> {
public:

    multiMSqueue(size_t poolsize = scalableMPMCqueue<MSqueue>::DEFAULT_POOL_SIZE) {
        if (! scalableMPMCqueue<MSqueue>::init(poolsize)) {
            std::cerr << "multiMSqueue init ERROR, abort....\n";
            abort();
        }
        
        for(size_t i=0;i<poolsize;++i)
            if (pool[i].init()<0) {
                std::cerr << "ERROR initializing MSqueue, abort....\n";
                abort();
            }
    }
};


#if defined(HAVE_CDSLIB)
    //
    // WARNING: Experimental code!!!!
    //
class multiMSqueueCDS: public scalableMPMCqueue< cds::queue::MSQueue<cds::gc::hzp_gc, void *>  > {
    typedef scalableMPMCqueue<cds::queue::MSQueue<cds::gc::hzp_gc, void *> > SQ;
public:    
      multiMSqueueCDS(size_t poolsize = SQ::DEFAULT_POOL_SIZE) {
          if (SQ::init(poolsize) < 0) {
              std::cerr << "multiMSqueue init ERROR, abort....\n";
              abort();
          }
      }
};
#endif // HAVE_CDSLIB

} // namespace

#endif /* __FF_MPMCQUEUE_HPP_ */
