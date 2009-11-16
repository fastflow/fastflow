/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_FARM_HPP_
#define _FF_FARM_HPP_
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */


#include <iostream>
#include <vector>
#include <lb.hpp>
#include <gt.hpp>
#include <node.hpp>

namespace ff {


template<typename lb_t=ff_loadbalancer, typename gt_t=ff_gatherer>
class ff_farm: public ff_node {
public:
    enum { DEF_MAX_NUM_WORKERS=64, DEF_BUFF_ENTRIES=512};

    typedef lb_t LoadBalancer_t;
    typedef gt_t Gatherer_t;

    ff_farm(int buffer_entries=DEF_BUFF_ENTRIES, 
            int max_num_workers=DEF_MAX_NUM_WORKERS):
        nworkers(0),
        buffer_entries(buffer_entries),
        max_nworkers(max_num_workers),
        emitter(NULL),collector(NULL),fallback(NULL),
        lb(new LoadBalancer_t(max_num_workers)),gt(NULL),
        workers(new ff_node*[max_num_workers]) {
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
        return gt->set_filter(collector);
    }

    // TODO
    int remove_worker() {  return -1;  }

    /* The collector output queue will be connected 
     * to the emitter input queue (feedback channel).
     */
    int wrap_around() {
        if (!gt) {
            error("FARM, wrap_around needs that the collector module is present\n");
            return -1;
        }
        if (create_input_buffer(buffer_entries)<0) {
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
            Barrier(cardinality());
        }
        
        for(int i=0;i<nworkers;++i) {
            if (workers[i]->create_input_buffer(buffer_entries)<0) return -1;
            if (gt) if (workers[i]->create_output_buffer(buffer_entries)<0) return -1;
            lb->register_worker(workers[i]);
            if (gt) gt->register_worker(workers[i]);
        }

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
        if (run()<0) return -1;           
        if (wait()<0) return -1;
        return 0;
    }
    
    int wait(/* timeval */ ) {
        int ret=0;
        if (lb->wait()<0) ret=-1;
        if (gt) if (gt->wait()<0) ret=-1;
        return ret;
    }; 

    int   cardinality() const { 
        int card=0;
        for(int i=0;i<nworkers;++i) 
            card += workers[i]->cardinality();
        
        return (card + 1 + (gt?1:0));
    }
    
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


    int set_output_buffer(SWSR_Ptr_Buffer * const o) {
        if (!gt) {
            error("FARM with no collector, cannot create set output buffer\n");
            return -1;
        }
        gt->set_out_buffer(o);
        return 0;
    }

protected:
    int nworkers;
    int buffer_entries;
    int max_nworkers;

    ff_node          *  emitter;
    ff_node          *  collector;
    ff_node          *  fallback;

    ff_loadbalancer  * lb;
    ff_gatherer      * gt;
    ff_node         ** workers;
};
 

} // namespace ff

#endif /* _FF_FARM_HPP_ */
