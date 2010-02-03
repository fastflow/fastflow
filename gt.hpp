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

class ff_gatherer: public ff_thread {

    enum {TICKS2WAIT=1000};

    template <typename T1, typename T2>  friend class ff_farm;
    friend class ff_pipeline;

protected:

    virtual void gather_task(void ** result) {
        //register int cnt=0;
        do {
            for(register int nextr=0;nextr<nworkers;++nextr) 
                if (workers[nextr]->get(result)) return;
            // FIX!!!!
            //if (sched_collector  && ++cnt>POP_CNT_COLL) { cnt=0; sched_yield();}
            //else 
            FFTRACE(lostpopticks+=TICKS2WAIT;++popwait);
            ticks_wait(TICKS2WAIT);
        } while(1);
	}

    void push(void * task) {
         while (! buffer->push(task)) {
                // if (ch->thxcore>1) {
             // if (++cnt>PUSH_POP_CNT) { sched_yield(); cnt=0;}
             //    else ticks_wait(TICKS2WAIT);
             //} else 
             FFTRACE(lostpushticks+=TICKS2WAIT;++pushwait);
             ticks_wait(TICKS2WAIT);
         }     
    }

    bool pop(void ** task) {
        //register int cnt = 0;       
        if (!get_out_buffer()) return false;
        while (! buffer->pop(task)) {
            ticks_wait(TICKS2WAIT);
        } 
        return true;
    }

    bool pop_nb(void ** task) {
        if (!get_out_buffer()) return false;
        return buffer->pop(task);
    }  



public:
    ff_gatherer(int max_num_workers):
        nworkers(0),max_nworkers(max_num_workers),
        filter(NULL),workers(new ff_node*[max_num_workers]),
        buffer(NULL) {}

    int set_filter(ff_node * f) { 
        if (filter) {
            error("GT, setting collector filter\n");
            return -1;
        }
        filter = f;
        return 0;
    }

    void set_out_buffer(SWSR_Ptr_Buffer * const buff) { buffer=buff;}

    SWSR_Ptr_Buffer * const get_out_buffer() const { return buffer;}

    int  register_worker(ff_node * w) {
        if (nworkers>=max_nworkers) {
            error("GT, max number of workers reached (max=%d)\n",max_nworkers);
            return -1;
        }
        workers[nworkers++]= w;
        return 0;
    }

    virtual void * svc(void *) {
        int stop=0;
        void * task = NULL;
        bool outpresent  = (get_out_buffer() != NULL);

        gettimeofday(&wtstart,NULL);
        do {
            task = NULL;
            gather_task(&task);
            if (task == (void *)FF_EOS) ++stop;
            else {
                if (filter)  task = filter->svc(task);

                if (!task) break; // if the filter returns NULL we exit immediatly
                else if (outpresent && (task != GO_ON)) push(task);
            }
        } while(stop<nworkers);

        if (outpresent) {
            // push EOS
            task = (void *)FF_EOS;
            push(task);
        }

        gettimeofday(&wtstop,NULL);
        return NULL;
    }

    virtual int svc_init() { 
        gettimeofday(&tstart,NULL);
        if (filter) return filter->svc_init(); 
        return 0;
    }

    virtual void svc_end() {
        if (filter) filter->svc_end();
        gettimeofday(&tstop,NULL);
    }

    int run(bool=false) {  
        if (this->spawn()<0) {
            error("GT, waiting GT thread\n");
            return -1; 
        }
        return 0;
    }

    virtual double ffTime() {
        return diffmsec(tstop,tstart);
    }

    virtual double wffTime() {
        return diffmsec(wtstop,wtstart);
    }

    virtual const struct timeval & getstarttime() const { return tstart;}
    virtual const struct timeval & getstoptime()  const { return tstop;}
    virtual const struct timeval & getwstartime() const { return wtstart;}
    virtual const struct timeval & getwstoptime() const { return wtstop;}

#if defined(TRACE_FASTFLOW)    
    virtual void ffStats(std::ostream & out) { 
        out << "Collector: "
            << "  work-time   : " << wffTime() << "\n"
            << "  n. tasks    : " << taskcnt   << "\n"
            << "  n. push lost: " << pushwait  << " (ticks=" << lostpushticks << ")" << "\n"
            << "  n. pop lost : " << popwait   << " (ticks=" << lostpopticks  << ")" << "\n";
    }
#endif

private:
    int               nworkers;
    int               max_nworkers;
    ff_node         * filter;
    ff_node        ** workers;
    SWSR_Ptr_Buffer * buffer;

    struct timeval tstart;
    struct timeval tstop;
    struct timeval wtstart;
    struct timeval wtstop;

#if defined(TRACE_FASTFLOW)
    unsigned long taskcnt;
    unsigned long lostpushticks;
    unsigned long pushwait;
    unsigned long lostpopticks;
    unsigned long popwait;
#endif
};

} // namespace ff

#endif /*  _FF_GT_HPP_ */
