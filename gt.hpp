/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_GT_HPP_
#define _FF_GT_HPP_
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

class ff_gatherer: public ff_node {

    enum {TICKS2WAIT=1000};

protected:

    void gather_task(void ** result) {
        //register int cnt=0;
        do {
            for(register int nextr=0;nextr<nworkers;++nextr) 
                if (workers[nextr]->pop(result)) return;
            // FIX!!!!
            //if (sched_collector  && ++cnt>POP_CNT_COLL) { cnt=0; sched_yield();}
            //else 
            ticks_wait(TICKS2WAIT);
        } while(1);
	}

public:
    ff_gatherer(int max_num_workers):
        nworkers(0),max_nworkers(max_num_workers),
        filter(NULL),workers(new ff_node*[max_num_workers]) {}

    int set_filter(ff_node & f) { 
        if (filter) {
            error("GT, setting collector filter\n");
            return -1;
        }
        filter = &f;
        return 0;
    }

    int  register_worker(ff_node * w) {
        if (nworkers>=max_nworkers) {
            error("GT, max number of workers reached (max=%d)\n",max_nworkers);
            return -1;
        }
        workers[nworkers++]=w;
        return 0;
    }

    virtual void * svc(void *) {
        int stop=0;
        void * task;
        do {
            task = NULL;
            gather_task(&task);
            if (task == (void *)FF_EOS) ++stop;
            else {
                void * result = filter->svc(task);
                if (!result) ++stop;
            }
        } while(stop<nworkers);

        return NULL;
    }

    virtual int svc_init(void *) { 
        return filter->svc_init(NULL);
    }

    virtual void svc_end() {
        filter->svc_end();
        //double t = 
        farmTime(STOP_TIME);
        //std::cerr << "Collector  time= " << t << "\n";
    }

    int run() {  
        if (this->spawn()<0) {
            error("GT, waiting GT thread, id = %d\n",get_my_id());
            return -1; 
        }
        return 0;
    }


private:
    int        nworkers;
    int        max_nworkers;
    ff_node *  filter;
    ff_node ** workers;
};

} // namespace ff

#endif /*  _FF_GT_HPP_ */
