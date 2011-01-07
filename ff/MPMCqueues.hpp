/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef __FF_MPMCQUEUE_HPP_ 
#define __FF_MPMCQUEUE_HPP_ 

/* 
 * This file contains Multi-Producer/Multi-Consumer queue implementations.
 *
 *
 *  -- Massimo Torquati <torquati@di.unipi.it> <massimotor@gmail.com>
 */


#include <stdlib.h>
#include <ff/buffer.hpp>
#include <ff/allocator.hpp>
#include <ff/atomic/abstraction_dcas.h>

namespace ff {


/* 
 * MSqueue is an implementation of the lock-free FIFO MPMC queue by
 * Maged M. Michael and Michael L. Scott described in the paper:
 * "Simple, Fast, and Practical Non-Blocking and Blocking
 * Concurrent Queue Algorithms", PODC 1996.
 *
 * The following implementation is inspired to the implementation
 * of the same algorithm from liblfds, a portable, license-free, 
 * lock-free data structure library written in C. 
 * More info about liblfds can be found at http://www.liblfds.org
 *
 *
 */
class MSqueue {
private:
    enum {MSQUEUE_PTR=0,MSQUEUE_CNT=1 };
#define DCAS abstraction_dcas
    struct Node;
    struct Pointer {
        Pointer() { ptr[MSQUEUE_PTR]=0, ptr[MSQUEUE_CNT]=0;}

        inline bool operator !() {
            return (ptr[MSQUEUE_PTR]==0);
        }
        inline Pointer& operator=(const Pointer & p) {
            ptr[MSQUEUE_PTR]=p.ptr[MSQUEUE_PTR], ptr[MSQUEUE_CNT]=p.ptr[MSQUEUE_CNT];
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
            return ((ptr[MSQUEUE_PTR]==r.ptr[MSQUEUE_PTR]) &&
                    (ptr[MSQUEUE_CNT]==r.ptr[MSQUEUE_CNT]));
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
        
        inline void set(Node & node, unsigned long tag) {
            ptr[MSQUEUE_PTR]=&node, ptr[MSQUEUE_CNT]=(Node*)tag;
        }

        inline void setTag(unsigned long tag) {
            ptr[MSQUEUE_CNT] = (Node *)tag;
        }
        
        inline unsigned long getTag() const {
            return (unsigned long)(ptr[MSQUEUE_CNT]);
        }

        inline void * getData() const { return ptr[MSQUEUE_PTR]->getData(); }

        Node * volatile ptr[2];
    } ALIGN(ALIGN_DOUBLE_POINTER);

    struct Node {
        Node():data(0) { next.ptr[MSQUEUE_PTR]=0;next.ptr[MSQUEUE_CNT]=0;}
        Node(void * data):data(data) {
            next.ptr[MSQUEUE_PTR]=0;next.ptr[MSQUEUE_CNT]=0;
        }
        
        inline operator atom_t * () const { return (atom_t *)next; }

        inline void setTag(unsigned long tag) {
            next.setTag(tag);
        }

        inline unsigned long getTag() const {
            return next.getTag();
        }

        inline void   setData(void * const d) { data=d;}
        inline void * getData() const { return data; }

        Pointer   next;
        void    * data;
    } ALIGN(ALIGN_DOUBLE_POINTER);

    Pointer  head;
    long padding1[longxCacheLine-(sizeof(Pointer)/sizeof(long))];
    Pointer  tail;
    long padding2[longxCacheLine-(sizeof(Pointer)/sizeof(long))];
    atomic_long_t aba_counter;
    long padding3[longxCacheLine-(sizeof(atomic_long_t)/sizeof(long))];

private:
    inline void allocnode(Pointer & p, void * data) {
        union { Node * p1; void * p2;} pn;
#if defined(USE_STD_ALLOCATOR)
        if(posix_memalign((void**)&pn.p2, ALIGN_DOUBLE_POINTER,sizeof(Node))!=0) {
            abort();
        }
#else
        if (FFAllocator::instance()->posix_memalign((void**)&pn.p2,ALIGN_DOUBLE_POINTER,sizeof(Node))!=0) {
            abort();
        }            
#endif
        new (pn.p2) Node(data);
        p.set(*pn.p1, atomic_long_inc_return(&aba_counter));
        pn.p1->setTag(atomic_long_inc_return(&aba_counter));
    }

    inline void deallocnode( Node * n) {
        n->~Node();
#if defined(USE_STD_ALLOCATOR)
        free(n);
#else
        FFAllocator::instance()->free(n);
#endif
    }

public:

    MSqueue& operator=(const MSqueue& v) { 
        head=v.head;
        tail=v.tail;
        atomic_long_set(&aba_counter, atomic_long_read((atomic_long_t*)&v.aba_counter));
        return *this;
    }

    /* initialize the MSqueue */
    bool init() {
        // reset the ABA counter
        atomic_long_set(&aba_counter,0);
        
        // create the first NULL node 
        // so the queue is never really empty
        Pointer dummy;
        allocnode(dummy,NULL);
        
        head = dummy;
        tail = dummy;
        return true;
    }

    // insert method, it never fails if data is not NULL
    inline bool push(void * const data) {
        if (!data) return false;

        bool done = false;

        ALIGN(ALIGN_DOUBLE_POINTER) Pointer tailptr;
        ALIGN(ALIGN_DOUBLE_POINTER) Pointer next;
        ALIGN(ALIGN_DOUBLE_POINTER) Pointer node;
        allocnode(node,data);

        do {
            tailptr = tail;
            next    = tailptr.getNodeNext();

            if (tailptr == tail) {
                if (!next) { // tail was pointing to the last node
                    node.setTag(next.getTag()+1);
                    done = DCAS((volatile atom_t *)(tailptr.getNodeNext()), 
                                (atom_t *)node, 
                                (atom_t *)next);
                } else {     // tail was not pointing to the last node
                    next.setTag(tailptr.getTag() + 1);
                    DCAS((volatile atom_t *)tail, (atom_t *)next, (atom_t *)tailptr);
                }
            }
        } while(!done);
        node.setTag(tailptr.getTag() + 1);
        DCAS((volatile atom_t *)tail, (atom_t *)node, (atom_t *) tailptr);
        return true;
    }
    
    // extract method, it returns false if the queue is empty
    inline bool  pop(void ** data) {        
        if (!data) return false;
        bool done = false;

        ALIGN(ALIGN_DOUBLE_POINTER) Pointer headptr;
        ALIGN(ALIGN_DOUBLE_POINTER) Pointer tailptr;
        ALIGN(ALIGN_DOUBLE_POINTER) Pointer next;

        do {
            headptr = head;
            tailptr = tail;
            next    = headptr.getNodeNext();

            if (head == headptr) {
                if (headptr.getNode() == tailptr.getNode()) {
                    if (!next) return false; // empty
                    next.setTag(tailptr.getTag() + 1);
                    DCAS((volatile atom_t *)tail, (atom_t *)next, (atom_t *)tailptr);
                } else {
                    *data = next.getData();
                    next.setTag(headptr.getTag() + 1);
                    done = DCAS((volatile atom_t *)head, (atom_t *)next, (atom_t *)headptr);
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
 *
 *
 */
template <typename Q=MSqueue>
class scalableMPMCqueue {
public:
    enum {DEFAULT_POOL_SIZE=4};

    scalableMPMCqueue(size_t poolsize=DEFAULT_POOL_SIZE):pool(poolsize),stop(true) {
        atomic_long_set(&enqueue,0);
        atomic_long_set(&dequeue,0);
    }
    
    bool init(size_t poolsize = DEFAULT_POOL_SIZE) {
        if (poolsize > pool.size()) {
            pool.resize(poolsize);
        }

        for(size_t i=0;i<pool.size();++i)
            if (!pool[i].init()) return false;

        stop=false;

        return true;
    }

    inline void stop_producing() { 
        for(size_t i=0;i<pool.size();++i)
            pool[i].push((void*)FF_EOS);        
        stop=true;
    }

    // insert method, it never fails if data is not NULL
    inline bool push(void * const data) {
        if (!data || stop) return false;
        register long q = atomic_long_inc_return(&enqueue) % pool.size();
        return pool[q].push(data);
    }

    // extract method, it returns false if the queue is empty
    inline bool  pop(void ** data) {      
        if (!data) return false;
        register long q = atomic_long_inc_return(&dequeue) % pool.size();
        bool r = pool[q].pop(data);
        if (stop && !r) { *data = (void*)FF_EOS;   return true; }
        return r;
    }

    inline bool empty() {
         for(size_t i=0;i<pool.size();++i)
             if (!pool[i].empty()) return false;
         return true;
    }
private:
    std::vector<Q> pool;
    atomic_long_t enqueue;
    atomic_long_t dequeue;
    bool          stop;
};



} // namespace

#endif /* __FF_MPMCQUEUE_HPP_ */
