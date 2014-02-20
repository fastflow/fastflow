/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \link
 *  \file parallel_for.hpp
 *  \ingroup high_level_patterns_shared_memory
 *
 *  \brief This file describes the parallel_for/parallel_reduce skeletons.
 */
 
#ifndef _FF_PARFOR_HPP_
#define _FF_PARFOR_HPP_
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


// #ifndef __INTEL_COMPILER
// // see http://www.stroustrup.com/C++11FAQ.html#11
// #if __cplusplus <= 199711L
// #error "parallel_for requires C++11 features"
// #endif
// #endif

#include <algorithm>
#include <vector>
#include <cmath>
#include <functional>
#include <ff/lb.hpp>
#include <ff/node.hpp>
#include <ff/farm.hpp>


#if defined(__ICC)
#define PRAGMA_IVDEP _Pragma("ivdep")
#else
#define PRAGMA_IVDEP
#endif

namespace ff {

    /* -------------------- Parallel For/Reduce Macros -------------------- */
    /* Usage example:
     *                              // loop parallelization using 3 workers
     *                              // and a minimum task grain of 2
     *                              wthread = 3;
     *                              grain = 2;
     *  for(int i=0;i<N;++i)        FF_PARFOR_BEGIN(for,i,0,N,1,grain,wthread) {
     *    A[i]=f(i)          ---->    A[i]=f(i);
     *                              } FF_PARFOR_END(for);
     * 
     *   parallel for + reduction:
     *     
     *  s=4;                         
     *  for(int i=0;i<N;++i)        FF_PARFORREDUCE_BEGIN(for,s,0,i,0,N,1,grain,wthread) {
     *    s*=f(i)            ---->    s*=f(i);
     *                              } FF_PARFORREDUCE_END(for,s,*);
     *
     *                          
     *                              FF_PARFOR_INIT(pf,maxwthread);
     *                              ....
     *  while(k<nTime) {            while(k<nTime) {
     *    for(int i=0;i<N;++i)        FF_PARFORREDUCE_START(pf,s,0,i,0,N,1,grain,wthread) {
     *      s*=f(i,k);       ---->       s*=f(i,k);
     *  }                             } FF_PARFORREDUCE_STOP(pf,s,*);
     *                             }
     *                             ....
     *
     *                             FF_PARFOR_DON(pf);
     *
     * 
     *  NOTE: inside the body of the PARFOR/PARFORREDUCE, it is possible to use the 
     *        '_ff_thread_id' const integer variable to identify the thread id 
     *        running the sequential portion of the loop.
     */

    /**
     *  name : of the parallel for
     *  idx  : iteration index
     *  begin: for starting point
     *  end  : for ending point
     *  step : for step
     *  chunk: chunk size
     *  nw   : n. of worker threads
     */
#define FF_PARFOR_BEGIN(name, idx, begin, end, step, chunk, nw)                   \
    ff_forall_farm<int> name(nw,true);                                            \
    name.setloop(begin,end,step,chunk,nw);                                        \
    auto F_##name = [&] (const long ff_start_##idx, const long ff_stop_##idx,     \
                         const int _ff_thread_id, const int) -> int  {            \
        PRAGMA_IVDEP                                                              \
        for(long idx=ff_start_##idx;idx<ff_stop_##idx;idx+=step) 

    /* This is equivalent to the above one except that the user has to define
     * the for loop in the range (ff_start_idx,ff_stop_idx(
     * This can be useful if you have to perform some actions before starting
     * the loop.
     */
#define FF_PARFOR2_BEGIN(name, idx, begin, end, step, chunk, nw)                  \
    ff_forall_farm<int> name(nw,true);                                            \
    name.setloop(begin,end,step,chunk, nw);                                       \
    auto F_##name = [&] (const long ff_start_##idx, const long ff_stop_##idx,     \
                         const int _ff_thread_id, const int) -> int  {            \
    /* here you have to define the for loop using ff_start/stop_idx  */


#define FF_PARFOR_END(name)                                                       \
    return 0;                                                                     \
    };                                                                            \
    {                                                                             \
      if (name.getnw()>1) {                                                       \
        name.setF(F_##name);                                                      \
        if (name.run_and_wait_end()<0) {                                          \
            error("running parallel for\n");                                      \
        }                                                                         \
      } else F_##name(name.startIdx(),name.stopIdx(),0,0);                        \
    }

    /* ---------------------------------------------- */

    /**
     *  name    : of the parallel for
     *  var     : variable on which the reduce operator is applied
     *  identity: the value such that var == var op identity 
     *  idx     : iteration index
     *  begin   : for starting point
     *  end     : for ending point
     *  step    : for step
     *  chunk   : chunk size
     *  nw      : n. of worker threads
     * 
     *  op      : reduce operation (+ * ....) 
     */
#define FF_PARFORREDUCE_BEGIN(name, var,identity, idx,begin,end,step, chunk, nw)  \
    ff_forall_farm<decltype(var)> name(nw,true);                                  \
    name.setloop(begin,end,step,chunk,nw);                                        \
    auto ovar_##name = var; auto idtt_##name =identity;                           \
    auto F_##name =[&](const long start, const long stop,const int _ff_thread_id, \
                       const decltype(var) _var) mutable ->  decltype(var) {      \
        auto var = _var;                                                          \
        PRAGMA_IVDEP                                                              \
          for(long idx=start;idx<stop;idx+=step)

#define FF_PARFORREDUCE_END(name, var, op)                                        \
          return var;                                                             \
        };                                                                        \
    {                                                                             \
        if (name.getnw()>1) {                                                     \
          name.setF(F_##name,idtt_##name);                                        \
          if (name.run_and_wait_end()<0) {                                        \
            error("running forall_##name\n");                                     \
          }                                                                       \
          var = ovar_##name;                                                      \
          for(size_t i=0;i<name.getnw();++i)  {                                   \
              var op##= name.getres(i);                                           \
          }                                                                       \
        } else {                                                                  \
          var = ovar_##name;                                                      \
          var op##= F_##name(name.startIdx(),name.stopIdx(),0,idtt_##name);       \
        }                                                                         \
    }

    /* ---------------------------------------------- */

    /* FF_PARFOR_START and FF_PARFOR_STOP have the same meaning of 
     * FF_PARFOR_BEGIN and FF_PARFOR_END but they have to be used in 
     * conjunction with  FF_PARFOR_INIT FF_PARFOR_END.
     *
     * The same is for FF_PARFORREDUCE_START/END.
     */
#define FF_PARFOR_INIT(name, nw)                                                  \
    ff_forall_farm<int> *name = new ff_forall_farm<int>(nw);


#define FF_PARFOR_DECL(name)       ff_forall_farm<int> * name
#define FF_PARFOR_ASSIGN(name,nw)  name=new ff_forall_farm<int>(nw)
#define FF_PARFOR_DONE(name)       name->stop(); name->wait(); delete name

#define FF_PARFORREDUCE_INIT(name, type, nw)                                      \
    ff_forall_farm<type> *name = new ff_forall_farm<type>(nw)

#define FF_PARFORREDUCE_DECL(name,type)      ff_forall_farm<type> * name
#define FF_PARFORREDUCE_ASSIGN(name,type,nw) name=new ff_forall_farm<type>(nw)
#define FF_PARFORREDUCE_DONE(name)           name->stop();name->wait();delete name

#define FF_PARFOR_START(name, idx, begin, end, step, chunk, nw)                   \
    name->setloop(begin,end,step,chunk,nw);                                       \
    auto F_##name = [&] (const long ff_start_##idx, const long ff_stop_##idx,     \
                         const int _ff_thread_id, const int) -> int  {            \
        PRAGMA_IVDEP                                                              \
        for(long idx=ff_start_##idx;idx<ff_stop_##idx;idx+=step) 

#define FF_PARFOR2_START(name, idx, begin, end, step, chunk, nw)                  \
    name->setloop(begin,end,step,chunk,nw);                                       \
    auto F_##name = [&] (const long ff_start_##idx, const long ff_stop_##idx,     \
                         const int _ff_thread_id, const int) -> int  {            \
    /* here you have to define the for loop using ff_start/stop_idx  */


#define FF_PARFOR_STOP(name)                                                      \
    return 0;                                                                     \
    };                                                                            \
    name->setF(F_##name);                                                         \
    if (name->getnw()>1) {                                                        \
      if (name->run_then_freeze(name->getnw())<0)                                 \
          error("running ff_forall_farm (name)\n");                               \
      name->wait_freezing();                                                      \
    } else F_##name(name->startIdx(),name->stopIdx(),0,0) 
    
#define FF_PARFORREDUCE_START(name, var,identity, idx,begin,end,step, chunk, nw)  \
    name->setloop(begin,end,step,chunk,nw);                                       \
    auto ovar_##name = var; auto idtt_##name =identity;                           \
    auto F_##name =[&](const long start, const long stop,const int _ff_thread_id, \
                       const decltype(var) _var) mutable ->  decltype(var) {      \
        auto var = _var;                                                          \
        PRAGMA_IVDEP                                                              \
        for(long idx=start;idx<stop;idx+=step) 

#define FF_PARFORREDUCE_STOP(name, var, op)                                       \
          return var;                                                             \
        };                                                                        \
        name->setF(F_##name,idtt_##name);                                         \
        if (name->getnw()>1) {                                                    \
          if (name->run_then_freeze(name->getnw())<0)                             \
              error("running ff_forall_farm (name)\n");                           \
          name->wait_freezing();                                                  \
          var = ovar_##name;                                                      \
          for(size_t i=0;i<name->getnw();++i)  {                                  \
              var op##= name->getres(i);                                          \
          }                                                                       \
        } else {                                                                  \
          var = ovar_##name;                                                      \
          var op##= F_##name(name->startIdx(),name->stopIdx(),0,idtt_##name);     \
        }


#define FF_PARFORREDUCE_F_STOP(name, var, F)                                      \
          return var;                                                             \
        };                                                                        \
          name->setF(F_##name,idtt_##name);                                       \
        if (name->getnw()>1) {                                                    \
          if (name->run_then_freeze(name->getnw())<0)                             \
              error("running ff_forall_farm (name)\n");                           \
          name->wait_freezing();                                                  \
          var = ovar_##name;                                                      \
          for(size_t i=0;i<name->getnw();++i)  {                                  \
             F(var,name->getres(i));                                              \
          }                                                                       \
        } else {                                                                  \
            var = ovar_##name;                                                    \
            F(var,F_##name(name->startIdx(),name->stopIdx(),0,idtt_##name));      \
        }

    /* ------------------------------------------------------------------- */



    // parallel for task, it represents a range (start,end( of indexes
struct forall_task_t {
    forall_task_t():start(0),end(0) {}
    forall_task_t(long start, long end):start(start),end(end) {}
    forall_task_t(const forall_task_t &t):start(t.start),end(t.end) {}
    forall_task_t & operator=(const forall_task_t &t) { start=t.start,end=t.end; return *this; }
    void set(long s, long e)  { start=s,end=e; }

    long start;
    long end;
};

//  used just to redefine losetime_in
class foralllb_t: public ff_loadbalancer {
protected:
    virtual inline void losetime_in() { 
        if ((int)(getnworkers())>=ncores) {
            //FFTRACE(lostpopticks+=(100*TICKS2WAIT);++popwait); // FIX
            ff_relax(0);
            return;
        }    
        //FFTRACE(lostpushticks+=TICKS2WAIT;++pushwait);
        PAUSE();            
    }
public:
    foralllb_t(size_t n):ff_loadbalancer(n),ncores(ff_numCores()) {}
    int getNCores() const { return ncores;}
private:
    const int ncores;
};

// parallel for/reduce  worker node
template<typename Tres>
class forallreduce_W: public ff_node {
    typedef std::function<Tres(const long,const long, const int, const Tres)> F_t;
protected:
    virtual inline void losetime_in(void) {
        //FFTRACE(lostpopticks+=ff_node::TICKS2WAIT; ++popwait); // FIX
        if (aggressive) PAUSE();
        else ff_relax(0);
    }
public:
    forallreduce_W(F_t F):aggressive(true),F(F) {}

    inline void* svc(void* t) {
        auto task = (forall_task_t*)t;
        res = F(task->start,task->end,get_my_id(),res);
        return t;
    }
    void setF(F_t _F, const Tres idtt, bool a=true) { 
        F=_F, res=idtt, aggressive=a;
    }
    Tres getres() const { return res; }
protected:
    bool aggressive;
    F_t  F;
    Tres res;
};

static inline bool data_cmp(std::pair<long,forall_task_t> &a,
                            std::pair<long,forall_task_t> &b) {
    return a.first<b.first;
}

// parallel for/reduce task scheduler
class forall_Scheduler: public ff_node {
protected:
    std::vector<bool> active;
    std::vector<std::pair<long,forall_task_t> > data;
    std::vector<forall_task_t> taskv;
protected:
    // initialize the data vector
    virtual inline size_t init_data(long start, long stop) {
        const long numtasks  = std::ceil((stop-start)/(double)_step);
        long totalnumtasks   = std::ceil(numtasks/(double)_chunk);
        long tt     = totalnumtasks;
        size_t ntxw = totalnumtasks / _nw;
        size_t r    = totalnumtasks % _nw;

        if (ntxw == 0 && r>=1) {
            ntxw = 1, r = 0;
        }
        data.resize(_nw);
        taskv.resize(8*_nw); // 8 is the maximum n. of jumps, see the heuristic below
        
        long end, t=0, e;
        for(size_t i=0;i<_nw && totalnumtasks>0;++i, totalnumtasks-=t) {
            t       = ntxw + ((r>1 && (i<r))?1:0);
            e       = start + (t*_chunk - 1)*_step + 1;
            end     = (e<stop) ? e : stop;
            data[i] = std::make_pair(t, forall_task_t(start,end));
            start   = (end-1)+_step;
        }

        if (totalnumtasks) {
            assert(totalnumtasks==1);
            // try to keep the n. of tasks per worker as smaller as possible
            if (ntxw > 1) data[_nw-1].first += totalnumtasks;
            else --tt;
            data[_nw-1].second.end = stop;
        } 

        // printf("init_data\n");
        // for(size_t i=0;i<_nw;++i) {
        //     printf("W=%d %ld <%ld,%ld>\n", i, data[i].first, data[i].second.start, data[i].second.end);
        // }
        // printf("totaltasks=%ld\n", tt);

        return tt;
    }    
public:
    forall_Scheduler(ff_loadbalancer* lb, long start, long stop, long step, long chunk, size_t nw):
        active(nw),lb(lb),_start(start),_stop(stop),_step(step),_chunk(std::max((long)1,chunk)),totaltasks(0),_nw(nw) {
        totaltasks = init_data(start,stop);
        assert(totaltasks>=1);
    }
    forall_Scheduler(ff_loadbalancer* lb, size_t nw):
        active(nw),lb(lb),_start(0),_stop(0),_step(1),_chunk(1),totaltasks(0),_nw(nw) {
        totaltasks = init_data(0,0);
        assert(totaltasks==0);
    }

    void* svc(void* t) {
        if (t==NULL) {
            if (totaltasks==0) return NULL;
            if ( (totaltasks/(double)_nw) <= 1.0 || (totaltasks==1) ) {
                for(size_t wid=0;wid<_nw;++wid)
                    if (data[wid].first) {
                        taskv[wid].set(data[wid].second.start, data[wid].second.end);
                        lb->ff_send_out_to(&taskv[wid], wid);
                    } 
                return NULL;
            }
            {
            size_t remaining = totaltasks;
            const long endchunk = (_chunk-1)*_step + 1;
            int jump = 0;
            bool skip1=false; // ,skip2=false,skip3=false; 
      moretask:
            for(size_t wid=0;wid<_nw;++wid) {
                if (data[wid].first) {
                    long start = data[wid].second.start;
                    long end   = (std::min)(start+endchunk, data[wid].second.end);
                    taskv[wid+jump].set(start, end);
                    lb->ff_send_out_to(&taskv[wid+jump], wid);
                    --remaining, --data[wid].first;
                    (data[wid].second).start = (end-1)+_step;  
                    active[wid]=true;
                } else {
                    // if we do not have task at the beginning the thread is terminated
                    lb->ff_send_out_to(EOS, wid);
                    active[wid]=false;
                    skip1=true; //skip2=skip3=true;
                }
            }
            // January 2014 (massimo): this heuristic maight not be the best option in presence 
            // of very high load imbalance between iterations. 
            // Update: removed skip2 and skip3 so that it is less aggressive !

            jump+=_nw;
            assert((jump / _nw) <= 8);
            // heuristic: try to assign more task at the very beginning
            if (!skip1 && totaltasks>=4*_nw)   { skip1=true; goto moretask;}

            //if (!skip2 && totaltasks>=64*_nw)  { skip1=false; skip2=true; goto moretask;}
            //if (!skip3 && totaltasks>=1024*_nw){ skip1=false; skip2=false; skip3=true; goto moretask;}

            return (remaining<=0)?NULL:GO_ON;
            }
        }
        if (--totaltasks <=0) return NULL;
        auto task = (forall_task_t*)t;
        const long endchunk = (_chunk-1)*_step + 1;
        const int wid = lb->get_channel_id();
        int id  = wid;

        if (data[id].first) {
        go:
            long start = data[id].second.start;
            long end   = std::min(start+endchunk, data[id].second.end);
            task->set(start, end);
            lb->ff_send_out_to(task, wid);
            --data[id].first;
            (data[id].second).start = (end-1)+_step;
            return GO_ON;
        }
        // finds the id with the highest number of tasks
        id = (std::max_element(data.begin(),data.end(),data_cmp) - data.begin());
        if (data[id].first) {
            if (data[id].first==1) goto go;

            // steal half of the tasks
            size_t q = data[id].first >> 1;
            size_t r = data[id].first & 0x1; 
            data[id].first  = q;
            data[wid].first = q+r;
            data[wid].second.end   = data[id].second.end;
            data[id].second.end    = data[id].second.start + _chunk*q;
            data[wid].second.start = data[id].second.end;
            id = wid;
            goto go;
        }
        if (active[wid]) {
            lb->ff_send_out_to(EOS, wid); // the thread is terminated
            active[wid]=false;
        }
        return GO_ON;
    }

    inline bool setloop(long start, long stop, long step, long chunk, size_t nw) {
        _start=start, _stop=stop, _step=step, _chunk=std::max((long)1,chunk), _nw=nw;
        totaltasks = init_data(start,stop);
        assert(totaltasks>=1);        
        // if we have only 1 task per worker, the scheduler exits immediatly so we have to wait all workers
        // otherwise is sufficient to wait only for the scheduler
        if ( (totaltasks/(double)_nw) <= 1.0 || (totaltasks==1) ) {
           _nw = totaltasks;
           return true;
        }
        return false;
    }

    inline long startIdx() const { return _start;}
    inline long stopIdx()  const { return _stop;}
    inline long stepIdx()  const { return _step;}
    inline size_t running() const { return _nw; }

protected:
    ff_loadbalancer *lb;
    long             _start,_stop,_step;  // for step
    long             _chunk;              // a chunk of indexes
    size_t           totaltasks;          // total n. of tasks
    size_t           _nw;                 // num. of workers
};


template <typename Tres>
class ff_forall_farm: public ff_farm<foralllb_t> {
    typedef std::function<Tres(const long,const long, const int, const Tres)> F_t;
protected:
    // allows to remove possible EOS still in the input/output queues 
    // of workers
    inline void resetqueues(const int _nw) {
        const svector<ff_node*> nodes = getWorkers();
        for(int i=0;i<_nw;++i) nodes[i]->reset();
    }
public:
    Tres t; // not used
    int numCores;
    ff_forall_farm(size_t maxnw,const bool skipwarmup=false):
        ff_farm<foralllb_t>(false,100*maxnw,100*maxnw,true,maxnw,true),waitall(true) {
        numCores = ((foralllb_t*const)getlb())->getNCores();
        std::vector<ff_node *> forall_w;
        auto donothing=[](const long,const long,const int,const Tres) -> int {
            return 0;
        };
        for(size_t i=0;i<maxnw;++i)
            forall_w.push_back(new forallreduce_W<Tres>(donothing));
        ff_farm<foralllb_t>::add_emitter(new forall_Scheduler(getlb(),maxnw));
        ff_farm<foralllb_t>::add_workers(forall_w);
        ff_farm<foralllb_t>::wrap_around();
        if (!skipwarmup) {
            if (ff_farm<foralllb_t>::run_then_freeze()<0) {
                error("running base forall farm\n");
            } else ff_farm<foralllb_t>::wait_freezing();
        }
    }
    inline int run_then_freeze(ssize_t nw_=-1) {
        const ssize_t nwtostart = (nw_ == -1)?getNWorkers():nw_;
        resetqueues(nwtostart);
        // the scheduler skips the first pop
        getlb()->skipfirstpop();
        return ff_farm<foralllb_t>::run_then_freeze(nwtostart);
    }
    int run_and_wait_end() {
        resetqueues(getNWorkers());
        return ff_farm<foralllb_t>::run_and_wait_end();
    }

    inline int wait_freezing() {
        if (waitall) return ff_farm<foralllb_t>::wait_freezing();
        return getlb()->wait_lb_freezing();
    }

    inline void setF(F_t  _F, const Tres idtt=(Tres)0) { 
        const size_t nw = getNWorkers();
        const svector<ff_node*> &nodes = getWorkers();
        const bool mode = (nw <= numCores)?true:false;
        for(size_t i=0;i<nw;++i) ((forallreduce_W<Tres>*)nodes[i])->setF(_F, idtt, mode);
    }
    inline void setloop(long begin,long end,long step,long chunk, size_t nw) {
        assert(nw<=getNWorkers());
        forall_Scheduler *sched = (forall_Scheduler*)getEmitter();
        waitall = sched->setloop(begin,end,step,chunk,nw);
    }
    // return the number of workers running or supposed to run
    inline size_t getnw() { return ((const forall_Scheduler*)getEmitter())->running(); }
    
    inline Tres getres(int i) {
        return  ((forallreduce_W<Tres>*)(getWorkers()[i]))->getres();
    }
    inline long startIdx(){ return ((const forall_Scheduler*)getEmitter())->startIdx(); }
    inline long stopIdx() { return ((const forall_Scheduler*)getEmitter())->stopIdx(); }
    inline long stepIdx() { return ((const forall_Scheduler*)getEmitter())->stepIdx(); }
protected:
    bool   waitall;
};

} // namespace ff

#endif /* _FF_PARFOR_HPP_ */
