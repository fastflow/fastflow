/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_LB_HPP_
#define _FF_LB_HPP_
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
#include <utils.hpp>
#include <node.hpp>

namespace ff {

class ff_loadbalancer: public ff_node {
    
    enum {TICKS2WAIT=500};

protected:
    
    void push_eof() {
        //register int cnt=0;
        void * eof = (void *)FF_EOS;
        for(register int i=0;i<nworkers;++i) {
            while(!workers[i]->push(eof)) {
                //if (sched_emitter && (++cnt>PUSH_CNT_EMIT)) { 
                // cnt=0;sched_yield();}
                //else 
                ticks_wait(TICKS2WAIT);
            }
        }	
    }

    virtual int schedule_task(void * task) {
        register int cnt,cnt2;
        do {
            cnt=0,cnt2=0;
            do {
                nextw = ++nextw % nworkers;
                if(workers[nextw]->push(task)) goto out;
                else if (++cnt == nworkers) break; 
            } while(1);
            if (fallback) {
                nextw=-1;
                //std::cerr << "exec fallback task= " << ((int*)task)[0] << "\n";
                fallback->svc(task);
                goto out;
            }
            else {
                // FIX!!!
                //if (sched_emitter && (++cnt2>PUSH_CNT_EMIT)) { cnt2=0; sched_yield();}
                //else 
                ticks_wait(TICKS2WAIT);
            }
        } while(1);
    out: ;
        return nextw;
    }


public:

    ff_loadbalancer(int max_num_workers):
        nworkers(0),max_nworkers(max_num_workers),nextw(0),
        filter(NULL),workers(new ff_node*[max_num_workers]),
        fallback(NULL),fallback_args(NULL) {}

    ~ff_loadbalancer() {
        if (workers) delete [] workers;
    }

    int set_filter(ff_node & f) { 
        if (filter) {
            error("LB, setting collector filter\n");
            return -1;
        }
        filter = &f;
        return 0;
    }

    int set_fallback(ff_node * fb, void * args) {
        if (fallback) {
            error("LB, setting fallback\n");
            return -1;
        }
        fallback = fb;
        fallback_args = args;

        return 0;
    }

    int  register_worker(ff_node * w) {
        if (nworkers>=max_nworkers) {
            error("LB, max number of workers reached (max=%d)\n",max_nworkers);
            return -1;
        }
        workers[nworkers++]= w;
        return 0;
    }

    int deregister_worker() {  // TODO

        return -1;
    }

    virtual void * svc(void *) {
        do {
            void * task = filter->svc(NULL);
            if (!task) break;
            schedule_task(task);
        } while(true);

        push_eof();
           
        return NULL;
    }

    virtual int svc_init(void *) { 
        double t = farmTime(START_TIME);

        std::cerr << "Emitter  time= " << t << "\n";

        if (filter->svc_init(NULL) <0) return -1;
        
        if (fallback) 
            fallback->svc_init(fallback_args);

        return 0;
    }

    virtual void svc_end(void * result) {
        filter->svc_end(result);
    }

    int run() {
        for(int i=0;i<nworkers;++i) {
            if (workers[i]->spawn()<0) {
                error("LB, spawning worker thread\n");
                return -1;
            }
        }
        if (this->spawn()<0) {
            error("LB, spawning LB thread\n");
            return -1;
        }
        return 0;
    }

    int wait() {
        int ret=0;
        for(int i=0;i<nworkers;++i)
            if (workers[i]->wait()<0) {
                error("LB, waiting worker thread, id = %d\n",workers[i]->get_my_id());
                ret = -1;
            }
        if (ff_thread::wait()<0) {
            error("LB, waiting LB thread, id = %d\n",get_my_id());
            ret = -1;
        }

        return ret;
    }


private:
    int         nworkers;
    int         max_nworkers;
    int         nextw;
    ff_node  *  filter;
    ff_node **  workers;
    ff_node  *  fallback;
    void     *  fallback_args;
};

} // namespace ff

#endif  /* _FF_LB_HPP_ */
