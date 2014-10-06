/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file barrier.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow blocking and non-blocking barrier implementations
 *
 *
 */

#ifndef FF_BARRIER_HPP
#define FF_BARRIER_HPP

/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

#include <stdlib.h>
#include <ff/platforms/platform.h>
#include <ff/utils.hpp>
#include <ff/config.hpp>


namespace ff {

    /**
     *  \class ffBarrier
     *  \ingroup building_blocks
     *
     *  \brief Just a barrier interface
     *  
     */
struct ffBarrier {
    virtual ~ffBarrier() {}
    virtual inline int  barrierSetup(size_t init) { return 0;}
    virtual inline void doBarrier(size_t) { };
};

    /**
     *  \class Barrier
     *  \ingroup building_blocks
     *
     *  \brief Blocking barrier - Used only to start all nodes synchronously
     *  
     */

#if (defined(__APPLE__) || defined(_MSC_VER))
class Barrier: public ffBarrier {
public:   
    Barrier(const size_t=MAX_NUM_THREADS):threadCounter(0),_barrier(0) {
        if (pthread_mutex_init(&bLock,NULL)!=0) {
            error("FATAL ERROR: Barrier: pthread_mutex_init fails!\n");
            abort();
        }
        if (pthread_cond_init(&bCond,NULL)!=0) {
            error("FATAL ERROR: Barrier: pthread_cond_init fails!\n");
            abort();
        }
    }
   
    inline int barrierSetup(size_t init) {
        assert(init>0);
        _barrier = init; 
        return 0;
    }

    inline void doBarrier(size_t) {
        pthread_mutex_lock(&bLock);
        if (!--_barrier) pthread_cond_broadcast(&bCond);
        else {
            pthread_cond_wait(&bCond, &bLock);
            assert(_barrier==0);
        }
        pthread_mutex_unlock(&bLock);
    }

    // TODO: better move counter methods in a different class
    inline size_t getCounter() const { return threadCounter;}
    inline void   incCounter()       { ++threadCounter;}
    inline void   decCounter()       { --threadCounter;}

private:
    // This is just a counter, and is used to set the ff_node::tid value.
    size_t            threadCounter;
    // it is the number of threads in the barrier. 
    size_t _barrier;
    pthread_mutex_t bLock;  // Mutex variable
    pthread_cond_t  bCond;  // Condition variable
};

#else

class Barrier: public ffBarrier {
public:
    Barrier(const size_t=MAX_NUM_THREADS):threadCounter(0),_barrier(0) { }
    ~Barrier() { if (_barrier>0) pthread_barrier_destroy(&bar); }

    inline int barrierSetup(size_t init) {
        assert(init>0);
        if (_barrier == init) return 0;
        if (_barrier==0) {
            if (pthread_barrier_init(&bar,NULL,init) != 0) {
                error("ERROR: pthread_barrier_init failed\n");
                return -1;
            }
            _barrier = init;
            return 0;
        }
        if (pthread_barrier_destroy(&bar) != 0) {
            error("ERROR: pthread_barrier_destroy failed\n");
            return -1;
        }
        if (pthread_barrier_init(&bar,NULL,init) == 0) {
            _barrier = init;
            return 0;
        }
        error("ERROR: pthread_barrier_init failed\n");
        return -1;
    }

    inline void doBarrier(size_t) {  
        pthread_barrier_wait(&bar); 
    }

    // TODO: better move counter methods in a different class
    inline size_t getCounter() const { return threadCounter;}
    inline void     incCounter()       { ++threadCounter;}
    inline void     decCounter()       { --threadCounter;}

private:
    // This is just a counter, and is used to set the ff_node::tid value.
    size_t            threadCounter;
    // it is the number of threads in the barrier. 
    size_t _barrier;
    pthread_barrier_t bar;
};

#endif 

/**
 *  \class spinBarrier
 *  \ingroup building_blocks
 *
 *  \brief Non-blocking barrier - Used only to start all nodes synchronously
 *
 */
class spinBarrier: public ffBarrier {
public:
   
    spinBarrier(const size_t maxNThreads=MAX_NUM_THREADS):_barrier(0),threadCounter(0),maxNThreads(maxNThreads) {
        atomic_long_set(&B[0],0);
        atomic_long_set(&B[1],0);
        barArray=new bool[maxNThreads];
        for(size_t i=0;i<maxNThreads;++i) barArray[i]=false;
    }

    ~spinBarrier() {
        if (barArray != NULL) delete [] barArray;
        barArray=NULL;
    }
    
    inline int barrierSetup(size_t init) {
        assert(init>0);
        for(size_t i=0; i<init; ++i) barArray[i]=false;
        _barrier = init; 
        return 0;
    }

    inline void doBarrier(size_t tid) {
        assert(tid<maxNThreads);
        const int whichBar = (barArray[tid] ^= true); // computes % 2
        long c = atomic_long_inc_return(&B[whichBar]);
        if ((size_t)c == _barrier) {
            atomic_long_set(&B[whichBar], 0);
            return;
        }
        // spin-wait
        while(c) { 
            c= atomic_long_read(&B[whichBar]);
            PAUSE();  // TODO: define a spin policy !
        }
    }

    // TODO: better move counter methods in a different class
    inline size_t getCounter() const { return threadCounter;}
    inline void   incCounter()       { ++threadCounter;}
    inline void   decCounter()       { --threadCounter;}
    
private:
    size_t _barrier;
    size_t threadCounter;
    const size_t maxNThreads;
    bool* barArray;          
    atomic_long_t B[2];
};


template<bool spin>
struct barHelper {
    Barrier bar;
    inline int barrierSetup(size_t init) { return bar.barrierSetup(init); }
    inline void doBarrier(size_t tid)    { return bar.doBarrier(tid); }
};
template<>
struct barHelper<true> {
    spinBarrier bar;
    inline int barrierSetup(size_t init) { return bar.barrierSetup(init); }
    inline void doBarrier(size_t tid)    { return bar.doBarrier(tid); }
};

/**
 *  \class barrierSelector
 *  \ingroup building_blocks
 *
 *  \brief It allows to select (at compile time) between blocking (false) and non-blocking (true) barriers.
 *
 */
template<bool whichone>
struct barrierSelector: public barHelper<whichone> {};

} // namespace ff

#endif /* FF_BARRIER_HPP */
