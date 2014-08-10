/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \file parallel_for.hpp
 *  \ingroup high_level_patterns
 *
 *  \brief This file describes the parallel_for/parallel_reduce skeletons.
 *  
 *  \detail Realize a parallel for loop. Different iterations should be 
 *  independent (data-races management is not autimatically provided
 *  but can be introduced by programmers via mutex).
 *
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
 *  please use the **parallel_for_static** method.
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
    
/*! 
  * \class ParallelFor
  *  \ingroup high_level_patterns
  * 
  * \brief Parallel for loop
  *
  * \detail Identifies an iterative work-sharing construct that specifies a region
  * (i.e. a Lambda function) in which the iterations of the associated loop 
  * should be executed in parallel. 
  * 
  * \example ../tests/parfor_basic.cpp
  */ 
class ParallelFor {
protected:
    ff_forall_farm<forallreduce_W<int> > *pf; 
public:
    /**
     * \brief Constructor

     * Set up a parallel for ParallelFor pattern run-time support 
     * (i.e. spawn workers threads)
     * A single object can be used as many times as needed to run different parallel for
     * pattern instances (different loop bodies). They cannot be nested nor recursive.
     * Nonblocking policy is to be preferred in case of repeated call of the 
     * some of the parallel_for methods (e.g. within a strict outer loop). On the same
     * ParallelFor object different parallel_for methods (e.g. parallel_for and 
     * parallel_for_thid, parallel_for_idx) can be called in sequence.

     * \param maxnw Maximum number of worker threads (not including active scheduler, if
     * any). Deafault <b>FF_AUTO</b> = N. of HW contexts. 
     * \param spinwait barrier kind. \p true nonblocking, \p false blocking.
     * Nonbloking barrier will leave worker threads active until class destruction is called 
     * (the threads will be active and in the nonblocking barrier only after the 
     * first call to one of the parallel_for methods). To put threads to sleep between different
     * calls, the <b>threadPause</b> method may be called.
     */
    ParallelFor(const long maxnw=FF_AUTO,bool spinwait=false):
        pf(new ff_forall_farm<forallreduce_W<int> >(maxnw,spinwait)) {}
    /**
     * \brief Destructor
     * 
     * \detail Terminate ParallelFor run-time support and makes resources housekeeping.
     * Both nonlocking and blocking worker threads are terminated.
     */
    ~ParallelFor()                { FF_PARFOR_DONE(pf); }

    /**
     * \brief Disable active scheduler (i.e. Emitter thread)
     *
     * \detail Disable active scheduler (i.e. Emitter thread of the master-worker
     * implementation). Active scheduling uses one dedicated nonblocking thread.
     * In passive scheduling, workers cooperatively schedule tasks via synchronisations
     * in memory. None of the above is always faster than the other: it depends on
     * parallelism degree, task grain and platform. 
     * As rule of thumb on large multicore and fine-grain tasks active scheduling is
     * faster. On few cores passive scheduler enhances overall performance.
     * Active scheduler is the default option.
     * \param onoff <b>true</b> disable active schduling, 
     * <b>false</b> enable active scheduling
     */ 

    inline void disableScheduler(bool onoff=true) { 
        pf->disableScheduler(onoff);
    }

    // It puts all spinning threads to sleep. It does not disable the spinWait flag
    // so at the next call, threads start spinning again.
    inline int threadPause() {
        return pf->stopSpinning();
    }

    // -------------------- parallel_for --------------------

    /**
     * \brief Parallel for region (basic) - static
     *
     * \detail Static scheduling onto nw worker threads.
     * Data is statically cut in maximal partitions, i.e.
     * partition size = last-first/nw
     * 
     * \param first first value of the iteration variable
     * \param last last value of the iteration variable
     * \param f <b>f(const long idx)</b>  Lambda function, 
     * body of the parallel loop. <b>idx</b>: iterator
     * \param nw number of worker threads (default FF_AUTO)
     */
    template <typename Function>
    inline void parallel_for(long first, long last, const Function& f, 
                             const long nw=FF_AUTO) {
        FF_PARFOR_START(pf, parforidx,first,last,1,PARFOR_STATIC(0),nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pf);
    }

    /**
     * \brief Parallel for region (step) - static
     *
     * \detail Static scheduling onto nw worker threads. 
     * Iteration space is walked with stride <b>step</b>. 
     * Data is statically cut in maximal partitions, i.e.
     * partition size = last-first/(nw*step)
     * 
     * \param first first value of the iteration variable
     * \param last last value of the iteration variable
     * \param step step increment for the iteration variable
     * \param f <b>f(const long idx)</b> body of the parallel loop
     * \param nw number of worker threads
     */
    template <typename Function>
    inline void parallel_for(long first, long last, long step, const Function& f, 
                             const long nw=FF_AUTO) {
        FF_PARFOR_START(pf, parforidx,first,last,step,PARFOR_STATIC(0),nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pf);
    }


    /**
     * @brief Parallel for region (step, grain) - dynamic  
     *
     * @detail Dynamic scheduling onto nw worker threads. Iterations are scheduled in 
     * blocks of minimal size <b>grain</b> 
     * Iteration space is walked with stride <b>step</b>. 
     * 
     * @param first first value of the iteration variable
     * @param last last value of the iteration variable
     * @param step step increment for the iteration variable
     * @param grain (> 0) minimum computation grain 
     * (n. of iterations scheduled together to a single worker)
     * @param f <b>f(const long idx)</b>  Lambda function, 
     * body of the parallel loop. <b>idx</b>: iteration
     * param nw number of worker threads
     */
    template <typename Function>
    inline void parallel_for(long first, long last, long step, long grain, 
                             const Function& f, const long nw=FF_AUTO) {
        FF_PARFOR_START(pf, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pf);
    }    

    /**
     * @brief Parallel for region with threadID (step, grain, thid) - dynamic
     *
     * @detail Dynamic scheduling onto nw worker threads. Iterations are scheduled in 
     * blocks of minimal size <b>grain</b> 
     * Iteration space is walked with stride <b>step</b>. <b>thid</b> Worker thread ID
     * is made available via a Lambda parameter.
     * 
     * @param first first value of the iteration variable
     * @param last last value of the iteration variable
     * @param step step increment for the iteration variable
     * @param grain  minimum computation grain  (n. of iterations scheduled together to a single worker)
     * @param f <b>f(const long idx, const int thid)</b>  Lambda function, body of the parallel loop. <b>idx</b>: iteration, <b>thid</b>: worker_id 
     * @param nw number of worker threads (default n. of platform HW contexts)
     */
    template <typename Function>
    inline void parallel_for_thid(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=FF_AUTO) {
        FF_PARFOR_START(pf, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(parforidx,_ff_thread_id);            
        } FF_PARFOR_STOP(pf);
    }    

    /**
     * @brief Parallel for region with indexes ranges (step, grain, thid, idx) - 
     * dynamic - advanced usage
     *
     * @detail Dynamic scheduling onto nw worker threads. Iterations are scheduled in 
     * blocks of minimal size <b>grain</b>. Iteration space is walked with stride 
     * <b>step</b>. A chunk of <b>grain</b> iterations are assigned to each worker but
     * they are not automatically walked. Each chunk can be traversed within the
     * parallel_for body (e.g. with a for loop within <b>f</b> with the same step). 
     *
     * \note Useful in few cases only - requires some expertise
     *
     * @param first first value of the iteration variable
     * @param last last value of the iteration variable
     * @param step step increment for the iteration variable
     * @param grain (> 0) minimum computation grain  (n. of iterations scheduled 
     * together to a single worker)
     * @param f <b>f(const long start_idx, const long stop_idx, const int thid)
     * </b>  Lambda function, body of the parallel loop. 
     * <b>start_idx</b> and <b>stop_idx</b>: iteration bounds assigned to 
     * worker_id <b>thid</b>. 
     * @param nw number of worker threads (default n. of platform HW contexts)
     */
    template <typename Function>
    inline void parallel_for_idx(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=FF_AUTO) {
        FF_PARFOR_START_IDX(pf,parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(ff_start_idx, ff_stop_idx,_ff_thread_id);            
        } FF_PARFOR_STOP(pf);
    }

    /**
     * @brief Parallel for region (step, grain) - static
     * 
     *
     * @detail Static scheduling onto nw worker threads. Iterations are scheduled in 
     * blocks of minimal size <b>grain > 1</b> or in maximal partitions 
     * <b>grain == 0</b>. Iteration space is walked with stride 
     * <b>step</b>. 
     *
     *
     * @param first first value of the iteration variable
     * @param last last value of the iteration variable
     * @param step step increment for the iteration variable
     * @param grain (> 0) minimum computation grain  (n. of iterations scheduled 
     * together to a single worker)
     * @param f <b>f(const long idx)</b>
     * Lambda function, body of the parallel loop. 
     * <b>start_idx</b> and <b>stop_idx</b>: iteration bounds assigned to 
     * worker_id <b>thid</b>. 
     * @param nw number of worker threads (default n. of platform HW contexts)
     */
    template <typename Function>
    inline void parallel_for_static(long first, long last, long step, long grain, 
                                    const Function& f, const long nw=FF_AUTO) {
        if (grain==0 || labs(nw)==1) {
            // Divide in evenly partioned parts
            FF_PARFOR_START(pf, parforidx,first,last,step,PARFOR_STATIC(grain),nw) {
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
    ParallelForReduce(const long maxnw=FF_AUTO, bool spinwait=false):
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
                             const long nw=FF_AUTO) {
        FF_PARFOR_T_START(pfr, T, parforidx,first,last,1,PARFOR_STATIC(0),nw) {
            f(parforidx);            
        } FF_PARFOR_T_STOP(pfr,T);
    }
    template <typename Function>
    inline void parallel_for(long first, long last, long step, const Function& f, 
                             const long nw=FF_AUTO) {
        FF_PARFOR_T_START(pfr, T, parforidx,first,last,step,PARFOR_STATIC(0),nw) {
            f(parforidx);            
        } FF_PARFOR_T_STOP(pfr,T);
    }
    template <typename Function>
    inline void parallel_for(long first, long last, long step, long grain, 
                             const Function& f, const long nw=FF_AUTO) {
        FF_PARFOR_T_START(pfr, T, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(parforidx);            
        } FF_PARFOR_STOP(pfr);
    }    

    template <typename Function>
    inline void parallel_for_thid(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=FF_AUTO) {
        FF_PARFOR_T_START(pfr,T, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(parforidx,_ff_thread_id);            
        } FF_PARFOR_T_STOP(pfr,T);
    }    

    template <typename Function>
    inline void parallel_for_idx(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=FF_AUTO) {

        FF_PARFOR_T_START_IDX(pfr,T, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            f(ff_start_idx, ff_stop_idx,_ff_thread_id);            
        } FF_PARFOR_T_STOP(pfr,T);
    }    

    template <typename Function>
    inline void parallel_for_static(long first, long last, long step, long grain, 
                                    const Function& f, const long nw=FF_AUTO) {
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
                                const long nw=FF_AUTO) {
        FF_PARFORREDUCE_START(pfr, var, identity, parforidx, first, last, 1, PARFOR_STATIC(0), nw) {
            body(parforidx, var);            
        } FF_PARFORREDUCE_F_STOP(pfr, var, finalreduce);
    }
    template <typename Function, typename FReduction>
    inline void parallel_reduce(T& var, const T& identity, 
                                long first, long last, long step, 
                                const Function& body, const FReduction& finalreduce,
                                const long nw=FF_AUTO) {
        FF_PARFORREDUCE_START(pfr, var, identity, parforidx,first,last,step,PARFOR_STATIC(0),nw) {
            body(parforidx, var);            
        } FF_PARFORREDUCE_F_STOP(pfr, var, finalreduce);
    }
    template <typename Function, typename FReduction>
    inline void parallel_reduce(T& var, const T& identity, 
                                long first, long last, long step, long grain, 
                                const Function& body, const FReduction& finalreduce,
                                const long nw=FF_AUTO) {
        FF_PARFORREDUCE_START(pfr, var, identity, parforidx,first,last,step,PARFOR_DYNAMIC(grain),nw) {
            body(parforidx, var);            
        } FF_PARFORREDUCE_F_STOP(pfr, var, finalreduce);
    }
    
    template <typename Function, typename FReduction>
    inline void parallel_reduce_static(T& var, const T& identity,
                                       long first, long last, long step, long grain, 
                                       const Function& body, const FReduction& finalreduce,
                                       const long nw=FF_AUTO) {
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
                                     const long nw=FF_AUTO) {
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
    ParallelForPipeReduce(const long maxnw=FF_AUTO, bool spinwait=false):
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
                                    const long nw=FF_AUTO) {
        
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
                                 const Function& Map, const long nw=FF_AUTO) {
        
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
                         const long nw=FF_AUTO) {
    FF_PARFOR_BEGIN(pfor, parforidx,first,last,1,PARFOR_STATIC(0),nw) {
        body(parforidx);            
    } FF_PARFOR_END(pfor);
}
//! Parallel loop over a range of indexes using a given step
template <typename Function>
static void parallel_for(long first, long last, long step, const Function& body, 
                         const long nw=FF_AUTO) {
    FF_PARFOR_BEGIN(pfor, parforidx,first,last,step,PARFOR_STATIC(0),nw) {
        body(parforidx);            
    } FF_PARFOR_END(pfor);
}
//! Parallel loop over a range of indexes using a given step and granularity
template <typename Function>
static void parallel_for(long first, long last, long step, long grain, 
                         const Function& body, const long nw=FF_AUTO) {
    FF_PARFOR_BEGIN(pfor, parforidx,first,last,step,grain,nw) {
        body(parforidx);            
    } FF_PARFOR_END(pfor);
}

template <typename Function, typename Value_t, typename FReduction>
void parallel_reduce(Value_t& var, const Value_t& identity, 
                     long first, long last, 
                     const Function& body, const FReduction& finalreduce,
                     const long nw=FF_AUTO) {
    Value_t _var = var;
    FF_PARFORREDUCE_BEGIN(pfr, _var, identity, parforidx, first, last, 1, PARFOR_STATIC(0), nw) {
        body(parforidx, _var);            
    } FF_PARFORREDUCE_F_END(pfr, _var, finalreduce);
    var=_var;
}
    
} // namespace ff

#endif /* FF_PARFOR_HPP */
    
