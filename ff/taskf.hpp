/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/*! 
 *  \link
 *  \file taskf.hpp
 *  \ingroup high_level_patterns
 *
 *  \brief This file implements a task parallel pattern whose tasks are functions.
 */
 
#ifndef FF_TASKF_HPP
#define FF_TASKF_HPP
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
 * Author: Massimo Torquati (September 2014)
 *
 */
#include <algorithm> 
#include <ff/farm.hpp>
#include <ff/task_internals.hpp>

namespace ff {

class ff_taskf {
    enum {DEFAULT_OUTSTANDING_TASKS = 2048};
protected:
    inline task_f_t *alloc_task(std::vector<param_info> &P, base_f_t *wtask) {
        task_f_t *task = &(TASKS[ntasks++ % outstandingTasks]);
        task->P     = P;
        task->wtask = wtask;
        return task;
    }
    
    /* --------------  worker ------------------------------- */
    struct Worker: ff_node_t<task_f_t> {
        inline task_f_t *svc(task_f_t *task) {
            task->wtask->call();
            return task;
        }
    };
    
    /* --------------  Scheduler ---------------------------- */
    class Scheduler: public ff_node_t<task_f_t> {
    protected:
        inline bool fromInput() { return (lb->get_channel_id() == -1);	}
    public:
        Scheduler(ff_loadbalancer*const lb, const int):
            eosreceived(false),numtasks(0), lb(lb) {}
        
        inline task_f_t *svc(task_f_t *task) { 
            if (fromInput()) { 
                ++numtasks; 
                return task;
            }
            if (--numtasks <= 0 && eosreceived) return EOS;
            return GO_ON; 
        }
        
        void thaw(bool freeze=false) { ff_node::thaw(freeze); };
        int wait_freezing() {  return lb->wait_lb_freezing(); }
        int wait()          {  return lb->wait(); }
        void eosnotify(ssize_t id) { 
            if (id == -1) { 
                eosreceived=true; 
                if (numtasks<=0) lb->broadcast_task(EOS);
            }
        }
    protected:	
        bool   eosreceived;
        size_t numtasks;
        
        ff_loadbalancer *const lb;
    };
public:
    // NOTE: by default the scheduling is round-robing (pseudo round-robin indeed).
    //       In order to select the ondemand scheduling policy, set the ondemand_buffer to a 
    //       value grather than 1.
    ff_taskf(int maxnw=ff_realNumCores(), const size_t maxTasks=DEFAULT_OUTSTANDING_TASKS, const int ondemand_buffer=0):
        farmworkers(maxnw),ntasks(0),outstandingTasks(std::max(maxTasks, (size_t)(MAX_NUM_THREADS*8))) {
        
        farm = new ff_farm<>(true,outstandingTasks,outstandingTasks,true,maxnw,true);
        TASKS.resize(outstandingTasks); 
        std::vector<ff_node *> w;
        // NOTE: Worker objects are going to be destroyed by the farm destructor
        for(int i=0;i<maxnw;++i) w.push_back(new Worker);
        farm->add_workers(w);
        farm->add_emitter(sched = new Scheduler(farm->getlb(), maxnw));
        farm->wrap_around(true);
        farm->set_scheduling_ondemand(ondemand_buffer);
        // FIX: scheduling on-demand 
        
        if (farm->run_then_freeze()<0) {
            error("ff_taskf: running farm\n");
        } else {
            farm->offload(EOS);
            farm->wait_freezing();
            farm->reset();
        }
    }
    virtual ~ff_taskf() {
        if (sched) delete sched;
        if (farm)  delete farm;
    }
    
    template<typename F_t, typename... Param>
    inline task_f_t* AddTask(const F_t F, Param... args) {	
        ff_task_f_t<F_t, Param...> *wtask = new ff_task_f_t<F_t, Param...>(F, args...);
        std::vector<param_info> useless;
        task_f_t *task = alloc_task(useless,wtask);	
        while(!farm->offload(task, 1)) ff_relax(1);	
        return task;
    } 
    
    virtual inline int run_and_wait_end() {
        while(!farm->offload(EOS, 1)) ff_relax(1);
        farm->thaw(true,farmworkers);
        sched->wait_freezing();
        return sched->wait();
    }
    virtual int run_then_freeze(ssize_t nw=-1) {
        while(!farm->offload(EOS, 1)) ff_relax(1);
        farm->thaw(true,farmworkers);
        int r=sched->wait_freezing();
        farm->reset();
        return r;
    }
    
    virtual inline int run()  { 
        farm->thaw(true,farmworkers); 
        return 0;
    }
    virtual inline int wait() { 
        while(!farm->offload(EOS, 1)) ff_relax(1);
        int r=sched->wait_freezing();
        farm->reset();
        return r;
    }
    
protected:
    int farmworkers;
    ff_farm<> *farm;
    Scheduler *sched;
    size_t ntasks, outstandingTasks;
    std::vector<task_f_t> TASKS;    // FIX: svector should be used here
};

} // namespace

#endif /* FF_TASKF_HPP */
