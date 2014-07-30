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
 *
 *  This file contains the ParallelFor and the ParallelForReduce classes 
 *  (and also some static functions).
 * 
 *  Iterations scheduling options:
 *
 *      1 - default static scheduling
 *      2 - static scheduling with grain size greater than 0
 *      3 - dynamic scheduling with grain size greater than 0
 * 
 *  As a general rule, the scheduling strategy is selected according to the chunk value:
 *      - chunk == 0 means default static scheduling, that is, ~(#iteration_space/num_workers) 
 *                   iterations per thread)
 *      - chunk >  0 means dynamic scheduling with grain equal to chunk, that is,
 *                   no more than chunk iterations at a time is computed by one thread, the 
 *                   chunk is assigned to workers thread dynamically
 *      - chunk <  0 means static scheduling with grain equal to chunk, that is,
 *                   the iteration space is divided into chunks each one of no more 
 *                   than chunk iterations. Then chunks are assigned to the workers threads 
 *                   statically and in a round-robin fashion.
 *
 *  If you want to use the static scheduling policy (either default or with a given grain),
 *  please use the parallel_for_static construct.
 *
 *  To use or not to use a scheduler thread ?
 *  As always, it depends on the application, scheduling strategy, platform at hand, 
 *  parallelism degree, ...etc....
 *
 *  The general rule is: a scheduler thread is started if:
 *   1. the dynamic scheduling policy is used (chunk>0);
 *   2. there are enough cores for hosting both worker threads and the scheduler thread;
 *   3. the number of tasks per thread is greater than 1.
 *
 *  In case of static scheduling (chunk <= 0), the scheduler thread is never started.
 *  It is possible to explicitly disable/enable the presence of the scheduler thread
 *  both at compile time and at run-time by using the disableScheduler method and the 
 *  two defines NO_PARFOR_SCHEDULER_THREAD and PARFOR_SCHEDULER_THREAD. 
 *
 *
 *  How to use the ParallelFor (in a nutshell) :
 *                                      ParallelForReduce<long> pfr;
 *    for(long i=0;i<N;i++)             pfr.parallel_for(0,N,[&](const long i) {
 *       A[i]=f(i);                         A[i]=f(i);
 *    long sum=0;               --->    });
 *    for(long i=0; i<N;++i)            long sum=0;
 *       sum+=g(A[i]);                  pfr.parallel_reduce(sum,0, 0,N,[&](const long i,long &sum) {
 *                                         sum+=g(A[i]);
 *                                      }, [](long &v, const long elem) {v+=elem;});
 *
 */

#ifndef FF_PARFOR_HPP
#define FF_PARFOR_HPP

#include <ff/pipeline.hpp>
#include <ff/parallel_for_internals.hpp>

namespace ff {

//
// TODO: to re-write the ParallelFor class as a specialization of the ParallelForReduce
//
    
//! ParallelFor class
class ParallelFor {
protected:
    ff_forall_farm<forallreduce_W<int> > *pf; 
public:
    ParallelFor(const long maxnw=-1,bool spinwait=false):
        pf(new ff_forall_farm<forallreduce_W<int> >(maxnw,spinwait)) {}

    ~ParallelFor()                { FF_PARFOR_DONE(pf); }

    // By calling this method with 'true' the scheduler will be disabled,
    // to restore the usage of the scheduler thread just pass 'false' as 
    // parameter
    inline void disableScheduler(bool onoff=true) { 
        pf->disableScheduler(onoff);
    }

    // It puts all spinning threads to sleep. It does not disable the spinWait flag
    // so at the next call, threads start spinning again.
    inline int threadPause() {
        return pf->stopSpinning();
    }

    /* -------------------- parallel_for -------------------- */
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
        FF_PARFOR_START(pf, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pf);
    }    
    template <typename Function>
    inline void parallel_for_thid(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=-1) {
        FF_PARFOR_START(pf, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(parforidx,_ff_thread_id);            
        } FF_PARFOR_STOP(pf);
    }    

    template <typename Function>
    inline void parallel_for_idx(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=-1) {
        FF_PARFOR_START_IDX(pf,parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(ff_start_idx, ff_stop_idx,_ff_thread_id);            
        } FF_PARFOR_STOP(pf);
    }

    template <typename Function>
    inline void parallel_for_static(long first, long last, long step, long grain, 
                                    const Function& f, const long nw=-1) {
        if (grain==0 || labs(nw)==1) {
            FF_PARFOR_START(pf, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
                f(parforidx);            
            } FF_PARFOR_STOP(pf);
        } else {
            FF_PARFOR_T_START_STATIC(pf, int, parforidx,first,last,step,PARFOR_STATIC(grain),nw) {
                f(parforidx);
            } FF_PARFOR_T_STOP(pf,int);
        }
    }
};

//! ParallelForReduce class
template<typename T>
class ParallelForReduce {
protected:
    ff_forall_farm<forallreduce_W<T> > * pfr; 

    // this constructor is useful to skip loop warmup and to disable spinwait
    ParallelForReduce(const long maxnw, bool spinWait, bool skipWarmup): 
        pfr(new ff_forall_farm<forallreduce_W<T> >(maxnw,false, true)) {}
public:
    ParallelForReduce(const long maxnw=-1, bool spinwait=false):
        pfr(new ff_forall_farm<forallreduce_W<T> >(maxnw,spinwait)) {}

    ~ParallelForReduce()                { FF_PARFORREDUCE_DONE(pfr); }

    // By calling this method with 'true' the scheduler will be disabled,
    // to restore the usage of the scheduler thread just pass 'false' as 
    // parameter
    inline void disableScheduler(bool onoff=true) { 
        pfr->disableScheduler(onoff);
    }

    // It puts all spinning threads to sleep. It does not disable the spinWait flag
    // so at the next call, threads start spinning again.
    inline int threadPause() {
        return pfr->stopSpinning();
    }

    /* -------------------- parallel_for -------------------- */
    template <typename Function>
    inline void parallel_for(long first, long last, const Function& f, 
                             const long nw=-1) {
        FF_PARFOR_T_START(pfr, T, parforidx,first,last,1,PARFOR_STATIC(0),nw) {
            f(parforidx);            
        } FF_PARFOR_T_STOP(pfr,T);
    }
    template <typename Function>
    inline void parallel_for(long first, long last, long step, const Function& f, 
                             const long nw=-1) {
        FF_PARFOR_T_START(pfr, T, parforidx,first,last,step,PARFOR_STATIC(0),nw) {
            f(parforidx);            
        } FF_PARFOR_T_STOP(pfr,T);
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
        } FF_PARFOR_T_STOP(pfr,T);
    }    

    template <typename Function>
    inline void parallel_for_idx(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=-1) {

        FF_PARFOR_T_START_IDX(pfr,T, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(ff_start_idx, ff_stop_idx,_ff_thread_id);            
        } FF_PARFOR_T_STOP(pfr,T);
    }    

    template <typename Function>
    inline void parallel_for_static(long first, long last, long step, long grain, 
                                    const Function& f, const long nw=-1) {
        if (grain==0 || labs(nw)==1) {
            FF_PARFOR_T_START(pfr, T, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
                f(parforidx);            
            } FF_PARFOR_T_STOP(pfr,T);
        } else {
            FF_PARFOR_T_START_STATIC(pfr, T, parforidx,first,last,step,PARFOR_STATIC(grain),nw) {
                f(parforidx);
            } FF_PARFOR_T_STOP(pfr,T);
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


#if defined(HAS_CXX11_VARIADIC_TEMPLATES)

//! ParallelForPipeReduce class
template<typename task_t>
class ParallelForPipeReduce {
protected:
    ff_forall_farm<forallpipereduce_W> *pfr; 
    struct reduceStage: ff_minode {        
        typedef std::function<void(const task_t &)> F_t;
        void *svc(void *t) {
            const task_t& task=reinterpret_cast<task_t>(t);
            F(task);
            return GO_ON;
        }
        int  wait() { return ff_minode::wait(); }

        void setF(F_t f) { F = f; }        
        F_t F;
    } reduce;
    ff_pipe<task_t>  pipe;

public:
    ParallelForPipeReduce(const long maxnw=-1, bool spinwait=false):
        pfr(new ff_forall_farm<forallpipereduce_W>(maxnw,false,true)), // skip loop warmup and disable spinwait
        pipe(pfr,&reduce) {
        
        // required to avoid error
        pfr->remove_collector();

        // avoiding initial barrier
        if (pipe.dryrun()<0)  // preparing all connections
            error("ParallelForPipeReduce: preparing pipe\n");
        
        // warmup phase
        pfr->resetskipwarmup();
        auto r=-1;
        if (pfr->run_then_freeze() != -1)         
            if (reduce.run_then_freeze() != -1)
                r = pipe.wait_freezing();            
        if (r<0) error("ParallelForPipeReduce: running pipe\n");


        if (spinwait) { // NOTE: spinning is enabled only for the Map part and not for the Reduce part
            if (pfr->enableSpinning() == -1)
                error("ParallelForPipeReduce: enabling spinwait\n");
        }
    }
    
    ~ParallelForPipeReduce()                { FF_PARFOR_DONE(pfr); reduce.wait(); }

    // By calling this method with 'true' the scheduler will be disabled,
    // to restore the usage of the scheduler thread just pass 'false' as 
    // parameter
    inline void disableScheduler(bool onoff=true) { 
        pfr->disableScheduler(onoff);
    }

    // It puts all spinning threads to sleep. It does not disable the spinWait flag
    // so at the next call, threads start spinning again.
    inline int threadPause() {
        return pfr->stopSpinning();
    }

    template <typename Function, typename FReduction>
    inline void parallel_reduce_idx(long first, long last, long step, long grain, 
                                    const Function& Map, const FReduction& Reduce,
                                    const long nw=-1) {
        
        pfr->setloop(first,last,step,grain,nw);
        pfr->setF(Map);
        reduce.setF(Reduce);
        auto r=-1;
        if (pfr->run_then_freeze(nw) != -1)
            if (reduce.run_then_freeze(nw) != -1)
                r = pipe.wait_freezing();            
        if (r<0) error("ParallelForPipeReduce: parallel_reduce_idx, starting pipe\n");
    }

    template <typename Function>
    inline void parallel_for_idx(long first, long last, long step, long grain, 
                                 const Function& Map, const long nw=-1) {
        
        pfr->setloop(first,last,step,grain,nw);
        pfr->setF(Map);
        auto r=-1;
        if (pfr->run_then_freeze(nw) != -1)
            r = pfr->wait_freezing();            
        if (r<0) error("ParallelForPipeReduce: parallel_for_idx, starting pipe\n");
    }


};
#endif 

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
    
