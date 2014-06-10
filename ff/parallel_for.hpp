/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \link
 *  \file parallel_for.hpp
 *  \ingroup high_level_patterns_shared_memory
 *
 *  \brief This file describes the parallel_for/parallel_reduce skeletons.
 */
 
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
/*
 *  - Author: 
 *     Massimo Torquati <torquati@di.unipi.it>
 *
 *  This file contains the ParallelFor and the ParallelForReduce classes 
 *  (and also some static functions).
 * 
 *  Iterations scheduling options:
 *
 *       - default static scheduling
 *       - static scheduling with grain size greater than 0
 *       - dynamic scheduling with grain size greater than 1
 * 
 *  How to use the ParallelFor:
 *                                      ParallelForReduce<long> pfr;
 *    for(long i=0;i<N;i++)             pfr.parallel_for(0,N,[&](const long i) {
 *       A[i]=f(i);                         A[i]=f(i);
 *    long sum=0;               --->    });
 *    for(long i=0; i<N;++i)            long sum=0;
 *       sum+=g(A[i]);                  pfr.parallel_reduce(sum,0, 0,N,[&](const long i,long &sum) {
 *                                         sum+=g(A[i]);
 *                                      }, [](long &v, const long elem) {v+=elem;});
 */

#ifndef FF_PARFOR_HPP
#define FF_PARFOR_HPP

#include <ff/parallel_for_internals.hpp>

namespace ff {

    
//! ParallelFor class
class ParallelFor {
protected:
    ff_forall_farm<int> * pf;
public:
    ParallelFor(const long maxnw=-1,bool spinwait=false, bool skipwarmup=false):
        pf(new ff_forall_farm<int>(maxnw,spinwait,skipwarmup)) {}

    ~ParallelFor()                { FF_PARFOR_DONE(pf); }

    // By calling this method with 'true' the scheduler will be disabled,
    // to restore the usage of the scheduler thread just pass 'false' as 
    // parameter
    inline void disableScheduler(bool onoff=true) { 
        pf->disableScheduler(onoff);
    }

    template <typename Function>
    inline void parallel_for(long first, long last, const Function& f, 
                             const long nw=-1) {
        FF_PARFOR_START(pf, parforidx,first,last,1,PARFOR_STATIC(0),nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pf);
    }
    template <typename Function>
    inline void parallel_for(long first, long last, long step, const Function& f, 
                             const long nw=-1) {
        FF_PARFOR_START(pf, parforidx,first,last,step,PARFOR_STATIC(0),nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pf);
    }
    template <typename Function>
    inline void parallel_for(long first, long last, long step, long grain, 
                             const Function& f, const long nw=-1) {
        FF_PARFOR_START(pf, parforidx,first,last,step,grain,nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pf);
    }    
    template <typename Function>
    inline void parallel_for_thid(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=-1) {
        FF_PARFOR_START(pf, parforidx,first,last,step,grain,nw) {
            f(parforidx,_ff_thread_id);            
        } FF_PARFOR_STOP(pf);
    }    

    template <typename Function>
    inline void parallel_for_idx(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=-1) {
        FF_PARFOR_START_IDX(pf,parforidx,first,last,step,grain,nw) {
            f(ff_start_idx, ff_stop_idx,_ff_thread_id);            
        } FF_PARFOR_STOP(pf);
    }

    template <typename Function>
    inline void parallel_for_static(long first, long last, long step, long grain, 
                                    const Function& f, const long nw=-1) {
        if (grain==0 || labs(nw)==1) {
            FF_PARFOR_START(pf, parforidx,first,last,step,grain,nw) {
                f(parforidx);            
            } FF_PARFOR_STOP(pf);
        } else {
            FF_PARFOR_T_START_STATIC(pf, int, parforidx,first,last,step,grain,nw) {
                f(parforidx);
            } FF_PARFOR_STOP(pf);
        }
    }
};

//! ParallelForReduce class
template<typename T>
class ParallelForReduce {
protected:
    ff_forall_farm<T> * pfr; 
public:
    ParallelForReduce(const long maxnw=-1, bool spinwait=false, bool skipwarmup=false):
        pfr(new ff_forall_farm<T>(maxnw,spinwait,skipwarmup)) {}

    ~ParallelForReduce()                { FF_PARFORREDUCE_DONE(pfr); }

    // By calling this method with 'true' the scheduler will be disabled,
    // to restore the usage of the scheduler thread just pass 'false' as 
    // parameter
    inline void disableScheduler(bool onoff=true) { 
        pfr->disableScheduler(onoff);
    }

    /* -------------------- parallel_for -------------------- */
    template <typename Function>
    inline void parallel_for(long first, long last, const Function& f, 
                             const long nw=-1) {
        FF_PARFOR_T_START(pfr, T, parforidx,first,last,1,PARFOR_STATIC(0),nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pfr);
    }
    template <typename Function>
    inline void parallel_for(long first, long last, long step, const Function& f, 
                             const long nw=-1) {
        FF_PARFOR_T_START(pfr, T, parforidx,first,last,step,PARFOR_STATIC(0),nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pfr);
    }
    template <typename Function>
    inline void parallel_for(long first, long last, long step, long grain, 
                             const Function& f, const long nw=-1) {
        FF_PARFOR_T_START(pfr, T, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pfr);
    }    

    template <typename Function>
    inline void parallel_for_thid(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=-1) {
        FF_PARFOR_T_START(pfr,T, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(parforidx,_ff_thread_id);            
        } FF_PARFOR_STOP(pfr);
    }    

    template <typename Function>
    inline void parallel_for_idx(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=-1) {

        FF_PARFOR_T_START_IDX(pfr,T, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(ff_start_idx, ff_stop_idx,_ff_thread_id);            
        } FF_PARFOR_STOP(pfr);
    }    

    template <typename Function>
    inline void parallel_for_static(long first, long last, long step, long grain, 
                                    const Function& f, const long nw=-1) {
        if (grain==0 || labs(nw)==1) {
            FF_PARFOR_T_START(pfr, T, parforidx,first,last,step,grain,nw) {
                f(parforidx);            
            } FF_PARFOR_STOP(pfr);
        } else {
            FF_PARFOR_T_START_STATIC(pfr, T, parforidx,first,last,step,PARFOR_STATIC(grain),nw) {
                f(parforidx);
            } FF_PARFOR_STOP(pfr);
        }
    }

    /* ------------------ parallel_reduce ------------------- */

    template <typename Function, typename FReduction>
    inline void parallel_reduce(T& var, const T& identity, 
                                long first, long last, 
                                const Function& body, const FReduction& finalreduce,
                                const long nw=-1) {
        FF_PARFORREDUCE_START(pfr, var, identity, parforidx, first, last, 1, PARFOR_STATIC(0), nw) {
            body(parforidx, var);            
        } FF_PARFORREDUCE_F_STOP(pfr, var, finalreduce);
    }
    template <typename Function, typename FReduction>
    inline void parallel_reduce(T& var, const T& identity, 
                                long first, long last, long step, 
                                const Function& body, const FReduction& finalreduce,
                                const long nw=-1) {
        FF_PARFORREDUCE_START(pfr, var, identity, parforidx,first,last,step,PARFOR_STATIC(0),nw) {
            body(parforidx, var);            
        } FF_PARFORREDUCE_F_STOP(pfr, var, finalreduce);
    }
    template <typename Function, typename FReduction>
    inline void parallel_reduce(T& var, const T& identity, 
                                long first, long last, long step, long grain, 
                                const Function& body, const FReduction& finalreduce,
                                const long nw=-1) {
        FF_PARFORREDUCE_START(pfr, var, identity, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            body(parforidx, var);            
        } FF_PARFORREDUCE_F_STOP(pfr, var, finalreduce);
    }
    
    template <typename Function, typename FReduction>
    inline void parallel_reduce_static(T& var, const T& identity,
                                       long first, long last, long step, long grain, 
                                       const Function& body, const FReduction& finalreduce,
                                       const long nw=-1) {
        if (grain==0 || labs(nw)==1) {
            FF_PARFORREDUCE_START(pfr, var, identity, parforidx,first,last,step,grain,nw) {
                body(parforidx, var);            
            } FF_PARFORREDUCE_F_STOP(pfr, var, finalreduce);
        } else {
            FF_PARFORREDUCE_START_STATIC(pfr, var, identity, parforidx,first,last,step,PARFOR_STATIC(grain),nw) {
                body(parforidx, var);
            } FF_PARFORREDUCE_F_STOP(pfr, var, finalreduce);
        }
    }
    
    template <typename Function, typename FReduction>
    inline void parallel_reduce_thid(T& var, const T& identity, 
                                     long first, long last, long step, long grain, 
                                     const Function& body, const FReduction& finalreduce,
                                     const long nw=-1) {
        FF_PARFORREDUCE_START(pfr, var, identity, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            body(parforidx, var, _ff_thread_id);            
        } FF_PARFORREDUCE_F_STOP(pfr, var, finalreduce);
    }
};


//
//---- static functions, useful for one-shot parallel for execution or when no extra settings are needed
//

//! Parallel loop over a range of indexes (step=1)
template <typename Function>
static void parallel_for(long first, long last, const Function& body, 
                         const long nw=-1) {
    FF_PARFOR_BEGIN(pfor, parforidx,first,last,1,PARFOR_STATIC(0),nw) {
        body(parforidx);            
    } FF_PARFOR_END(pfor);
}
//! Parallel loop over a range of indexes using a given step
template <typename Function>
static void parallel_for(long first, long last, long step, const Function& body, 
                         const long nw=-1) {
    FF_PARFOR_BEGIN(pfor, parforidx,first,last,step,PARFOR_STATIC(0),nw) {
        body(parforidx);            
    } FF_PARFOR_END(pfor);
}
//! Parallel loop over a range of indexes using a given step and granularity
template <typename Function>
static void parallel_for(long first, long last, long step, long grain, 
                         const Function& body, const long nw=-1) {
    FF_PARFOR_BEGIN(pfor, parforidx,first,last,step,grain,nw) {
        body(parforidx);            
    } FF_PARFOR_END(pfor);
}

template <typename Function, typename Value_t, typename FReduction>
void parallel_reduce(Value_t& var, const Value_t& identity, 
                     long first, long last, 
                     const Function& body, const FReduction& finalreduce,
                     const long nw=-1) {
    Value_t _var = var;
    FF_PARFORREDUCE_BEGIN(pfr, _var, identity, parforidx, first, last, 1, PARFOR_STATIC(0), nw) {
        body(parforidx, _var);            
    } FF_PARFORREDUCE_F_END(pfr, _var, finalreduce);
    var=_var;
}
    
    
} // namespace ff

#endif /* FF_PARFOR_HPP */
    
