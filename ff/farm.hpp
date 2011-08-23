/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_FARM_HPP_
#define _FF_FARM_HPP_
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

/*! \mainpage Fastflow main page
 
	\section intro_sec Introduction
 
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <ff/platforms/platform.h>
#include <ff/lb.hpp>
#include <ff/gt.hpp>
#include <ff/node.hpp>


namespace ff {


template<typename lb_t=ff_loadbalancer, typename gt_t=ff_gatherer>
class ff_farm: public ff_node {
protected:
    inline int   cardinality(Barrier * const barrier)  { 
        int card=0;
        for(int i=0;i<nworkers;++i) 
            card += workers[i]->cardinality(barrier);
        
        lb->set_barrier(barrier);
        if (gt) gt->set_barrier(barrier);

        return (card + 1 + (gt?1:0));
    }

    inline int prepare() {
        for(int i=0;i<nworkers;++i) {
            if (workers[i]->create_input_buffer((ondemand ? ondemand: (in_buffer_entries/nworkers + 1)))<0) return -1;
            if (gt || lb->masterworker()) 
                if (workers[i]->create_output_buffer(out_buffer_entries/nworkers + DEF_IN_OUT_DIFF)<0) 
                    return -1;
            lb->register_worker(workers[i]);
            if (gt) gt->register_worker(workers[i]);
        }
        prepared=true;
        return 0;
    }

    int freeze_and_run(bool=false) {
        freeze();
        return run(true);
    } 

public:
    enum { DEF_MAX_NUM_WORKERS=64, DEF_IN_BUFF_ENTRIES=2048, DEF_IN_OUT_DIFF=128, 
           DEF_OUT_BUFF_ENTRIES=(DEF_IN_BUFF_ENTRIES+DEF_IN_OUT_DIFF)};

    typedef lb_t LoadBalancer_t;
    typedef gt_t Gatherer_t;

    /* input_ch = true to set accelerator mode
     * in_buffer_entries = input queue length
     * out_buffer_entries = output queue length
     * max_num_workers = highest number of farm's worker
     * worker_cleanup = true deallocate worker object at exit
     */
    ff_farm(bool input_ch=false,
            int in_buffer_entries=DEF_IN_BUFF_ENTRIES, 
            int out_buffer_entries=DEF_OUT_BUFF_ENTRIES,
            bool worker_cleanup=false,
            int max_num_workers=DEF_MAX_NUM_WORKERS):
        has_input_channel(input_ch),prepared(false),ondemand(0),
        nworkers(0),
        in_buffer_entries(in_buffer_entries),
        out_buffer_entries(out_buffer_entries),
        worker_cleanup(worker_cleanup),
        max_nworkers(max_num_workers),
        emitter(NULL),collector(NULL),fallback(NULL),
        lb(new LoadBalancer_t(max_num_workers)),gt(NULL),
        workers(new ff_node*[max_num_workers]) {
        for(int i=0;i<max_num_workers;++i) workers[i]=NULL;

        if (has_input_channel) { 
            if (create_input_buffer(in_buffer_entries, false)<0) {
                error("FARM, creating input buffer\n");
            }
        }
    }
    
    ~ff_farm() { 
        if (lb) delete lb; 
        if (gt) delete(gt); 
        if (workers) {
            if (worker_cleanup) {
                for(int i=0;i<max_nworkers; ++i) 
                    if (workers[i]) delete workers[i];
            }
            delete [] workers;
        }
    }
    
    int add_emitter(ff_node * e, ff_node * fb=NULL) { 
        if (emitter) return -1; 

        /* NOTE: if there is a collector filter then no 
         * fallback execution is possible 
         */
        if ((collector || lb->masterworker()) && fb) {
            error("FARM, cannot add fallback function if the collector is present or master-worker configuration has been set\n");
            return -1;
        }

        emitter = e;         
        fallback=fb;
        if (lb->set_filter(emitter)) return -1;
        return lb->set_fallback(fallback);
    }

    /* The default scheduling policy is round-robin,
     * When there is a great computational difference among tasks
     * the round-robin scheduling policy could lead to load imbalance
     * in worker's workload (expecially with short stream length).
     * The on-demand scheduling policy can guarantee a near optimal
     * load balancing in lots of cases.
     *
     * Alternatively it is always possible to define a complete 
     * application-level scheduling by redefining the ff_loadbalancer class.
     *
     * The function parameter, sets the number of queue slot for 
     * one worker threads.
     *
     */
    void set_scheduling_ondemand(const int inbufferentries=1) { 
        if (in_buffer_entries<=0) ondemand=1;
        else ondemand=inbufferentries;
    }

    int add_workers(std::vector<ff_node *> & w) { 
        if ((nworkers+w.size())> (unsigned int)max_nworkers) {
            error("FARM, try to add too many workers, please increase max_nworkers\n");
            return -1; 
        }
        if ((nworkers+w.size())==0) {
            error("FARM, try to add zero workers!\n");
            return -1; 
        }        
        for(unsigned int i=nworkers;i<(nworkers+w.size());++i) {
            workers[i] = w[i];
            workers[i]->set_id(i);
        }
        nworkers+= (unsigned int) w.size();
        return 0;
    }

    int add_collector(ff_node * c, bool outpresent=false) { 
        if (collector) return -1; 

        if (fallback) {
            error("FARM, cannot add collector filter with fallback function\n");
            return -1;
        }
        
        if (!gt) gt = new Gatherer_t(max_nworkers);
        collector = c; 
        
        if (has_input_channel) { /* it's an accelerator */
            if (create_output_buffer(out_buffer_entries)<0) return -1;
        }
        
        return gt->set_filter(collector);
    }
    
    /* If the collector is present, than the collector output queue 
     * will be connected to the emitter input queue (feedback channel),
     * otherwise the emitter acts as collector filter (pure master-worker
     * skeleton).
     */
    int wrap_around() {
        if (fallback) {
            error("FARM, cannot add feedback channels if the fallback function has been set in the Emitter\n");
            return -1;
        }
        if (!gt) {
            if (lb->set_masterworker()<0) return -1;
            if (!has_input_channel) lb->skipfirstpop();
            return 0;
        }

        if (create_input_buffer(in_buffer_entries, false)<0) {
            error("FARM, creating input buffer\n");
            return -1;
        }
        
        if (set_output_buffer(get_in_buffer())<0) {
            error("FARM, setting output buffer\n");
            return -1;
        }

        lb->skipfirstpop();
        return 0;
    }

    int run(bool skip_init=false) {
        if (!skip_init) {
            // set the initial value for the barrier 

            //Barrier::instance()->barrier(cardinality());
            if (!barrier)  barrier = new Barrier;
            barrier->doBarrier(cardinality(barrier));
        }
        
        if (!prepared) if (prepare()<0) return -1;

        if (gt && gt->run()<0) {
            error("FARM, running gather module\n");
            return -1;
        }
        if (lb->run()<0) {
            error("FARM, running load-balancer module\n");
            return -1;        
        }
        return 0;
    }

    int run_and_wait_end() {
        if (isfrozen()) return -1; // FIX !!!!

        stop();
        if (run()<0) return -1;           
        if (wait()<0) return -1;
        return 0;
    }

    int run_then_freeze() {
        if (isfrozen()) {
            thaw();
            freeze();
            return 0;
        }
        if (!prepared) if (prepare()<0) return -1;
        freeze();
        return run();
    }
    int wait(/* timeval */ ) {
        int ret=0;
        if (lb->wait()<0) ret=-1;
        if (gt) if (gt->wait()<0) ret=-1;
        return ret;
    }

    int wait_freezing(/* timeval */ ) {
        int ret=0;
        if (lb->wait_freezing()<0) ret=-1;
        if (gt) if (gt->wait_freezing()<0) ret=-1;
        return ret; 
    } 

    void stop() {
        lb->stop();
        if (gt) gt->stop();
    }

    void freeze() {
        lb->freeze();
        if (gt) gt->freeze();
    }

    void thaw() {
        lb->thaw();
        if (gt) gt->thaw();
    }

    /* check if the farm is frozen */
    bool isfrozen() { return lb->isfrozen(); }

    /* offload the given task to the farm
     */
    inline bool offload(void * task,
                        unsigned int retry=((unsigned int)-1),
                        unsigned int ticks=ff_loadbalancer::TICKS2WAIT) { 
        FFBUFFER * inbuffer = get_in_buffer();

        if (inbuffer) {
            for(unsigned int i=0;i<retry;++i) {
                if (inbuffer->push(task)) return true;
                ticks_wait(ticks);
            }     
            return false;
        }
        
        if (!has_input_channel) 
            error("FARM: accelerator is not set, offload not available");
        else
            error("FARM: input buffer creation failed");
        return false;

    }    

    // return values:
    //   false: EOS arrived or too many retries
    //   true:  there is a new value
    inline bool load_result(void ** task,
                            unsigned int retry=((unsigned int)-1),
                            unsigned int ticks=ff_gatherer::TICKS2WAIT) {
        if (!gt) return false;
        for(unsigned int i=0;i<retry;++i) {
            if (gt->pop_nb(task)) {
                if ((*task != (void *)FF_EOS)) return true;
                else return false;
            }
            ticks_wait(ticks);
        }
        return false;
    }

    // return values:
    //   false: no task present
    //   true : there is a new value, you should check if the task is an FF_EOS
    inline bool load_result_nb(void ** task) {
        if (!gt) return false;
        return gt->pop_nb(task);
    }
    
    inline lb_t * const getlb() const { return lb;}

    /* the returned time comprise the time spent in svn_init and 
     * in svc_end methods
     */
    double ffTime() {
        if (gt)
            return diffmsec(gt->getstoptime(),
                            lb->getstarttime());

        const struct timeval zero={0,0};
        std::vector<struct timeval > workertime(nworkers+1,zero);
        for(int i=0;i<nworkers;++i)
            workertime[i]=workers[i]->getstoptime();
        workertime[nworkers]=lb->getstoptime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return diffmsec((*it),lb->getstarttime());
    }

    /*  the returned time considers only the time spent in the svc
     *  methods
     */
    double ffwTime() {
        if (gt)
            return diffmsec(gt->getwstoptime(),
                            lb->getwstartime());

        const struct timeval zero={0,0};
        std::vector<struct timeval > workertime(nworkers+1,zero);
        for(int i=0;i<nworkers;++i)
            workertime[i]=workers[i]->getwstoptime();
        workertime[nworkers]=lb->getwstoptime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return diffmsec((*it),lb->getwstartime());
    }


#if defined(TRACE_FASTFLOW)
    void ffStats(std::ostream & out) { 
        out << "--- farm:\n";
        lb->ffStats(out);
        for(int i=0;i<nworkers;++i) workers[i]->ffStats(out);
        if (gt) gt->ffStats(out);
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }

#endif
    
protected:

    // ff_node interface
    void* svc(void * task) { return NULL; }
    int   svc_init()       { return -1; };
    void  svc_end()        {}

    int create_input_buffer(int nentries, bool fixedsize) {
        if (in) {
            error("FARM create_input_buffer, buffer already present\n");
            return -1;
        }
        if (ff_node::create_input_buffer(nentries, fixedsize)<0) return -1;
        lb->set_in_buffer(in);
        return 0;
    }
    
    int create_output_buffer(int nentries, bool fixedsize=false) {
        if (!gt) {
            error("FARM with no collector, cannot create output buffer\n");
            return -1;
        }        
        if (out) {
            error("FARM create_output_buffer, buffer already present\n");
            return -1;
        }
        if (ff_node::create_output_buffer(nentries, fixedsize)<0) return -1;
        gt->set_out_buffer(out);
        return 0;
    }


    int set_output_buffer(FFBUFFER * const o) {
        if (!gt) {
            error("FARM with no collector, cannot set output buffer\n");
            return -1;
        }
        gt->set_out_buffer(o);
        return 0;
    }


protected:
    bool has_input_channel; // for accelerator
    bool prepared;
    int ondemand;          // if >0, emulates on-demand scheduling
    int nworkers;
    int in_buffer_entries;
    int out_buffer_entries;
    bool worker_cleanup;
    int max_nworkers;

    ff_node          *  emitter;
    ff_node          *  collector;
    ff_node          *  fallback;

    lb_t             * lb;
    ff_gatherer      * gt;
    ff_node         ** workers;
};
 

} // namespace ff

#endif /* _FF_FARM_HPP_ */
