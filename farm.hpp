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
#include <lb.hpp>
#include <gt.hpp>
#include <node.hpp>

namespace ff {

enum { DEF_MAX_NUM_WORKERS=64, DEF_BUFF_ENTRIES=512};

template<typename W_t, typename lb_t=ff_loadbalancer, typename gt_t=ff_gatherer>
class ff_farm {
public:
    typedef W_t  Worker_t;
    typedef lb_t LoadBalancer_t;
    typedef gt_t Gatherer_t;

    ff_farm(int nworkers, int buffer_entries=DEF_BUFF_ENTRIES, 
            int max_num_workers=DEF_MAX_NUM_WORKERS):
        nworkers(nworkers),buffer_entries(buffer_entries),
        max_nworkers(max_num_workers),
        emitter(NULL),collector(NULL),
        worker_args(NULL),
        lb(new LoadBalancer_t(max_num_workers)),gt(NULL),
        workers(new thWorker<Worker_t>*[max_num_workers]) {
    }
    
    ~ff_farm() { 
        if (lb) delete lb; 
        if (gt) delete(gt); 
        if (workers) {
            for(int i=0;i<nworkers;++i) 
                if (workers[i]) delete workers[i];
            delete [] workers;
        }        
    }
    
    void set_worker_args(void * args) { worker_args=args; }
    
    int add_emitter(ff_node & e, ff_node * fb=NULL, void * args=NULL) { 
        if (emitter) return -1; 
        // if there is a collector filter, no fallback execution is possible 
        if (collector && fb) return -1;
        emitter = &e;         
        fallback=fb;
        if (lb->set_filter(e)) return -1;
        return lb->set_fallback(fallback, args);
    }

    int add_collector(ff_node & c) { 
        if (collector) return -1; 
        // if there is a collector filter, no fallback execution is possible 
        if (fallback) return -1;

        if (!gt) gt = new Gatherer_t(max_nworkers);
        collector = &c; 
        return gt->set_filter(c);
    }


    int remove_worker() {
        // TODO
        return -1;
    }

    int run() {
        // init barrier
        Barrier(nworkers + 1 + (gt?1:0));

        for(int i=0;i<nworkers;++i) {
            workers[i] = new thWorker<Worker_t>(worker_args);
            workers[i]->create_input_buffer(buffer_entries); 
            workers[i]->create_output_buffer(buffer_entries); 
            workers[i]->set_id(i);
            lb->register_worker(workers[i]);
            if (gt) gt->register_worker(workers[i]);
        }

        if (gt && gt->run()<0) return -1;
        if (lb->run()<0) return -1;        

        return 0;
    }

    int run_and_wait_end() {
        if (run()<0) return -1;           
        lb->wait();
        if (gt) gt->wait();
        return 0;
    }
    
    int wait(/* timeval */ ) {
         // TODO
        return -1;
    }; 


protected:
    int nworkers;
    int buffer_entries;
    int max_nworkers;

    ff_node *  emitter;
    ff_node *  collector;
    ff_node *  fallback;

    void    *  worker_args;

    ff_loadbalancer        * lb;
    ff_gatherer            * gt;
    thWorker<Worker_t>    ** workers;
};
 

} // namespace ff

#endif /* _FF_FARM_HPP_ */
