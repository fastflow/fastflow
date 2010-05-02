/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_GT_HPP_
#define _FF_GT_HPP_
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
#include <utils.hpp>
#include <node.hpp>

namespace ff {

class ff_gatherer: public ff_thread {

    enum {TICKS2WAIT=5000};

    template <typename T1, typename T2>  friend class ff_farm;
    friend class ff_pipeline;

protected:

    inline int getnworkers() const { return nworkers;}

    /* return values: -1 no worker selected
     *                [0..numworkers[ the number of worker selected
     */
    virtual inline int selectworker() { return (++nextr % nworkers); }


    virtual inline void losetime_in() { 
        //FFTRACE(lostpopticks+=TICKS2WAIT;++popwait);
        //ticks_wait(TICKS2WAIT);
        
        FFTRACE(register ticks t0 = getticks());
        usleep(TICKS2WAIT);
        FFTRACE(register ticks diff=(getticks()-t0));
        FFTRACE(lostpopticks+=diff;++popwait);
    }

    /* try to collect one task from the pool of workers */
    virtual void gather_task(void ** result) {
        register int i=0;
        do {
            do {
                nextr=selectworker();
                if (workers[nextr]->get(result)) return;
            } while(i++<nworkers);

            losetime_in();
            i=0;
        } while(1);
	}

    void push(void * task) {
         while (! buffer->push(task)) {
                // if (ch->thxcore>1) {
             // if (++cnt>PUSH_POP_CNT) { sched_yield(); cnt=0;}
             //    else ticks_wait(TICKS2WAIT);
             //} else 
             //FFTRACE(lostpushticks+=TICKS2WAIT;++pushwait);
             //ticks_wait(TICKS2WAIT);
             FFTRACE(register ticks t0 = getticks());
             usleep(TICKS2WAIT);
             FFTRACE(register ticks diff=(getticks()-t0));
             FFTRACE(lostpushticks+=diff;++pushwait);
         }     
    }

    bool pop(void ** task) {
        //register int cnt = 0;       
        if (!get_out_buffer()) return false;
        while (! buffer->pop(task)) {
            //ticks_wait(TICKS2WAIT);
            usleep(TICKS2WAIT);
        } 
        return true;
    }

    bool pop_nb(void ** task) {
        if (!get_out_buffer()) return false;
        return buffer->pop(task);
    }  



public:
    ff_gatherer(int max_num_workers):
        nworkers(0),max_nworkers(max_num_workers),nextr(0),
        filter(NULL),workers(new ff_node*[max_num_workers]),
        buffer(NULL) {
        time_setzero(tstart);time_setzero(tstop);
        time_setzero(wtstart);time_setzero(wtstop);
        wttime=0;
        FFTRACE(taskcnt=0;lostpushticks=0;pushwait=0;lostpopticks=0;popwait=0;ticksmin=(ticks)-1;ticksmax=0;tickstot=0);
        for(int i=0;i<max_num_workers;++i) workers[i]=NULL;
    }

    ~ff_gatherer() { 
        if (workers) delete [] workers;
    }

    int set_filter(ff_node * f) { 
        if (filter) {
            error("GT, setting collector filter\n");
            return -1;
        }
        filter = f;
        return 0;
    }

    void set_out_buffer(FFBUFFER * const buff) { buffer=buff;}

    FFBUFFER * const get_out_buffer() const { return buffer;}

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
                FFTRACE(++taskcnt);
                if (filter)  {
                    FFTRACE(register ticks t0 = getticks());

                    task = filter->svc(task);

#if defined(TRACE_FASTFLOW)
                    register ticks diff=(getticks()-t0);
                    tickstot +=diff;
                    ticksmin=std::min(ticksmin,diff);
                    ticksmax=std::max(ticksmax,diff);
#endif    

                }

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
        wttime+=diffmsec(wtstop,wtstart);
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
            << "  work-time (ms): " << wttime    << "\n"
            << "  n. tasks      : " << taskcnt   << "\n"
            << "  svc ticks     : " << tickstot  << " (min= " << (filter?ticksmin:0) << " max= " << ticksmax << ")\n"
            << "  n. push lost  : " << pushwait  << " (ticks=" << lostpushticks << ")" << "\n"
            << "  n. pop lost   : " << popwait   << " (ticks=" << lostpopticks  << ")" << "\n";
    }
#endif

private:
    int               nworkers;
    int               max_nworkers;
    int               nextr;
    ff_node         * filter;
    ff_node        ** workers;
    FFBUFFER        * buffer;

    struct timeval tstart;
    struct timeval tstop;
    struct timeval wtstart;
    struct timeval wtstop;
    double wttime;

#if defined(TRACE_FASTFLOW)
    unsigned long taskcnt;
    ticks         lostpushticks;
    unsigned long pushwait;
    ticks         lostpopticks;
    unsigned long popwait;
    ticks         ticksmin;
    ticks         ticksmax;
    ticks         tickstot;
#endif
};

} // namespace ff

#endif /*  _FF_GT_HPP_ */
