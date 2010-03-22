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

#include <iostream>
#include <vector>
#include <lb.hpp>
#include <gt.hpp>
#include <node.hpp>

#include <algorithm>

namespace ff {


template<typename lb_t=ff_loadbalancer, typename gt_t=ff_gatherer>
class ff_farm: public ff_node {
protected:
    inline int   cardinality() const { 
        int card=0;
        for(int i=0;i<nworkers;++i) 
            card += workers[i]->cardinality();
        
        return (card + 1 + (gt?1:0));
    }

    inline int prepare() {
        for(int i=0;i<nworkers;++i) {
            if (workers[i]->create_input_buffer((ondemand?1:(in_buffer_entries/nworkers + 1)))<0) return -1;
            if (gt || lb->masterworker()) 
                if (workers[i]->create_output_buffer(out_buffer_entries/nworkers + DEF_IN_OUT_DIFF)<0) 
                    return -1;
            lb->register_worker(workers[i]);
            if (gt) gt->register_worker(workers[i]);
        }
        prepared=true;
        return 0;
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
     */
    ff_farm(bool input_ch=false,
            int in_buffer_entries=DEF_IN_BUFF_ENTRIES, 
            int out_buffer_entries=DEF_OUT_BUFF_ENTRIES, 
            int max_num_workers=DEF_MAX_NUM_WORKERS):
        has_input_channel(input_ch),prepared(false),ondemand(false),
        nworkers(0),
        in_buffer_entries(in_buffer_entries),
        out_buffer_entries(out_buffer_entries),
        max_nworkers(max_num_workers),
        emitter(NULL),collector(NULL),fallback(NULL),
        lb(new LoadBalancer_t(max_num_workers)),gt(NULL),
        workers(new ff_node*[max_num_workers]) {      
  
        if (has_input_channel) { 
            if (create_input_buffer(in_buffer_entries)<0) {
                error("FARM, creating input buffer\n");
            }
        }
    }
    
    ~ff_farm() { 
        if (lb) delete lb; 
        if (gt) delete(gt); 
        if (workers) delete [] workers;
    }
    
    int add_emitter(ff_node * e, ff_node * fb=NULL) { 
        if (emitter) return -1; 

        /* NOTE: if there is a collector filter then no 
         * fallback execution is possible 
         */
        if (collector && fb) {
            error("FARM, cannot add fallback function if the collector is present\n");
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
     */
    void set_scheduling_ondemand() { ondemand=true;}


    int add_workers(std::vector<ff_node *> & w) { 
        if ((nworkers+w.size())> (unsigned int)max_nworkers) {
            error("FARM, try to add too many workers, please increase max_nworkers\n");
            return -1; 
        }
        for(unsigned int i=nworkers;i<(nworkers+w.size());++i) {
            workers[i] = w[i];
            workers[i]->set_id(i);
        }
        nworkers+=w.size();
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
        if (!gt) {
            if (lb->set_masterworker()<0) return -1;
            if (!has_input_channel) lb->skipfirstpop();
            return 0;
        }

        if (create_input_buffer(in_buffer_entries)<0) {
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
            Barrier::instance()->barrier(cardinality());
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
            while (! inbuffer->push(task)) {
                ticks_wait(ff_loadbalancer::TICKS2WAIT);
            }     
            return true;
        }
        
        if (!has_input_channel) 
            error("Farm: accelerator is not set, offload not available");
        else
            error("Farm: input buffer creation failed");
        return false;

    }    

    // return values:
    //   false: EOS arrived
    //   true:  there is a new value
    inline bool load_result(void ** task /* TODO: timeout */) {
        if (!gt) return false;
        if (gt->pop(task) && (*task != (void *)FF_EOS)) return true;
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

    int create_input_buffer(int nentries) {
        if (in) {
            error("FARM create_input_buffer, buffer already present\n");
            return -1;
        }
        if (ff_node::create_input_buffer(nentries)<0) return -1;
        lb->set_in_buffer(in);
        return 0;
    }
    
    int create_output_buffer(int nentries) {
        if (!gt) {
            error("FARM with no collector, cannot create output buffer\n");
            return -1;
        }        
        if (out) {
            error("FARM create_output_buffer, buffer already present\n");
            return -1;
        }
        if (ff_node::create_output_buffer(nentries)<0) return -1;
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
    bool ondemand;          // if true, emulate on-demand scheduling
    int nworkers;
    int in_buffer_entries;
    int out_buffer_entries;
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
