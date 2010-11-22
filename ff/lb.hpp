/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_LB_HPP_
#define _FF_LB_HPP_
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
#include <deque>

#include <ff/utils.hpp>
#include <ff/node.hpp>

namespace ff {

class ff_loadbalancer: public ff_thread {
    
    enum {TICKS2WAIT=1000};

protected:
    inline bool push_task(void * task, int idx) {
        return workers[idx]->put(task);
    }    
    inline void push_eos(bool nofreeze=false) {
        //register int cnt=0;
        void * eos = (void *)(nofreeze?FF_EOS_NOFREEZE:FF_EOS);
        for(register int i=0;i<nworkers;++i) {
            while(!workers[i]->put(eos)) {
                //if (sched_emitter && (++cnt>PUSH_CNT_EMIT)) { 
                // cnt=0;sched_yield();}
                //else 
                losetime_out();
            }
        }	
    }

    inline int getnworkers() const { return nworkers;}
    inline ff_node * const getfallback() { return fallback;}

    /* The following functions can be redefined to implement new
     * scheduling policy.
     */

    /* return values: -1 no worker selected
     *                [0..numworkers[ the number of worker selected
     */
    virtual inline int selectworker() { return (++nextw % nworkers); }

    /* number of tentative before wasting some times and than retry */
    virtual inline int ntentative() { return nworkers;}

    virtual inline void losetime_out() { 
        //FFTRACE(lostpushticks+=TICKS2WAIT;++pushwait);
        //ticks_wait(TICKS2WAIT); 
        FFTRACE(register ticks t0 = getticks());
        usleep(TICKS2WAIT);
        FFTRACE(register ticks diff=(getticks()-t0));
        FFTRACE(lostpushticks+=diff;++pushwait);
    }

    virtual inline void losetime_in() { 
        FFTRACE(lostpopticks+=TICKS2WAIT;++popwait);
        ticks_wait(TICKS2WAIT); 
#if 0
        FFTRACE(register ticks t0 = getticks());
        usleep(TICKS2WAIT);
        FFTRACE(register ticks diff=(getticks()-t0));
        FFTRACE(lostpopticks+=diff;++popwait);
#endif
    }

    /* main scheduling function also called by ff_send_out! */
    virtual int schedule_task(void * task) {
        register int cnt,cnt2;
        do {
            cnt=0,cnt2=0;
            do {
                nextw = selectworker();
                if(workers[nextw]->put(task)) {
                    FFTRACE(++taskcnt);
                    return nextw;
                }
                else {
                    //std::cerr << ".";
                    if (++cnt == ntentative()) break; 
                }
            } while(1);
            if (fallback) {
                nextw=-1;
                //std::cerr << "exec fallback\n";
                fallback->svc(task);
                return nextw;
            }
            else losetime_out();
            //std::cerr << "-";
        } while(1);
        return nextw;
    }

    
    virtual std::deque<ff_node *>::iterator  collect_task(void ** task, 
                                                           std::deque<ff_node *> & availworkers,
                                                           std::deque<ff_node *>::iterator & start) {
        register int cnt, nw=(availworkers.end()-availworkers.begin());
        const std::deque<ff_node *>::iterator & ite(availworkers.end());
        do {
            cnt=0;
            do {
                if (++start == ite) start=availworkers.begin();
                if((*start)->get(task)) {
                    channelid = (*start)->get_my_id();
                    return start;
                }
                else if (++cnt == nw) {
                    // FIX: check!
                    if (buffer && buffer->pop(task)) {
                        channelid = -1;
                        return ite;
                    }
                    break;
                }
            } while(1);
            losetime_in();
        } while(1);
        return ite;
    }

    /* it sends the same task to all workers    
     *
     */
    virtual void broadcast_task(void * task) {
        std::vector<int> retry;

        for(register int i=0;i<nworkers;++i) {
            if(!workers[i]->put(task))
                retry.push_back(i);
        }
        while(retry.size()) {
            if(workers[retry.back()]->put(task))
                retry.pop_back();
            else losetime_out();
        }
    }
    
    bool pop(void ** task) {
        //register int cnt = 0;       
        while (! buffer->pop(task)) {
            //    if (ch->thxcore>1) {
            //if (++cnt>PUSH_POP_CNT) { sched_yield(); cnt=0;}
            //else ticks_wait(TICKS2WAIT);
            //} else 
            losetime_in();
        } 
        return true;
    }
    
    static bool ff_send_out_emitter(void * task,unsigned int retry,unsigned int ticks, void *obj) {
        ((ff_loadbalancer *)obj)->schedule_task(task);
        return true;
    }


public:

    ff_loadbalancer(int max_num_workers): 
        nworkers(0),max_nworkers(max_num_workers),nextw(0),nextINw(0),channelid(-1),
        filter(NULL),workers(new ff_node*[max_num_workers]),
        fallback(NULL),buffer(NULL),skip1pop(false),master_worker(false) {
        time_setzero(tstart);time_setzero(tstop);
        time_setzero(wtstart);time_setzero(wtstop);
        wttime=0;
        FFTRACE(taskcnt=0;lostpushticks=0;pushwait=0;lostpopticks=0;popwait=0;ticksmin=(ticks)-1;ticksmax=0;tickstot=0);
    }

    ~ff_loadbalancer() {
        if (workers) delete [] workers;
    }

    int set_filter(ff_node * f) { 
        if (filter) {
            error("LB, setting collector filter\n");
            return -1;
        }
        filter = f;
        filter->registerCallback(ff_send_out_emitter, this);

        return 0;
    }

    int set_fallback(ff_node * fb) {
        if (fallback) {
            error("LB, setting fallback\n");
            return -1;
        }
        fallback = fb;

        return 0;
    }

    void set_in_buffer(FFBUFFER * const buff) { buffer=buff;}

    FFBUFFER * const get_in_buffer() const { return buffer;}
    
    /* return the channel id of the last pop 
     *  -1 is the emitter input buffer
     */
    const int get_channel_id() const { return channelid;}
    void reset_channel_id() { channelid=-1;}

    void skipfirstpop() { skip1pop=true;}
    
    int  set_masterworker() {
        if (master_worker) {
            error("LB, master_worker flag already set\n");
            return -1;
        }
        master_worker=true;
        return 0;
    }

    const bool masterworker() const { return master_worker;}
    
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
        void * task = NULL;
        void * ret  = (void *)FF_EOS;
        bool inpresent  = (get_in_buffer() != NULL);
        bool skipfirstpop = skip1pop;

        gettimeofday(&wtstart,NULL);
        if (!master_worker) {
            do {
                if (inpresent) {
                    if (!skipfirstpop) pop(&task);
                    else skipfirstpop=false;
                    
                    if (task == (void*)FF_EOS) {
                        push_eos(); 
                        break;
                    } else if (task == (void*)FF_EOS_NOFREEZE) {
                        push_eos(true);
                        ret = task;
                        break;
                    }
                }
                
                if (filter) {
                    FFTRACE(register ticks t0 = getticks());

                    task = filter->svc(task);

#if defined(TRACE_FASTFLOW)
                    register ticks diff=(getticks()-t0);
                    tickstot +=diff;
                    ticksmin=std::min(ticksmin,diff);
                    ticksmax=std::max(ticksmax,diff);
#endif  
                    if (task == GO_ON) continue;

                    // if the filter returns NULL we exit immediatly
                    if (task ==(void*)FF_EOS_NOFREEZE) { 
                        push_eos(true); 
                        ret = task;
                        break; 
                    }
                    if (!task || task==(void*)FF_EOS) {
                        push_eos();
                        ret = (void*)FF_EOS;
                        break;
                    }
                } 
                
                schedule_task(task);
            } while(true);
        } else {
            int nw=nworkers;

            // contains current worker
            std::deque<ff_node *> availworkers; 
            for(int i=0;i<nworkers;++i)
                availworkers.push_back(workers[i]);

            std::deque<ff_node *>::iterator start(availworkers.begin());
            std::deque<ff_node *>::iterator victim(availworkers.begin());
            do {
                if (!skipfirstpop) {  
                    victim=collect_task(&task, availworkers, start);                    
                } else skipfirstpop=false;
                
                if ((task == (void*)FF_EOS) || 
                    (task == (void*)FF_EOS_NOFREEZE)) {
                    if (victim == availworkers.end())  push_eos((task==(void*)FF_EOS_NOFREEZE));
                    else {
                        availworkers.erase(victim);
                        start=availworkers.begin(); // restart iterator
                        --nw;
                    }
                    if (!nw) {
                        ret = task;
                        break; // received all EOS, exit
                    }
                } else {
                    if (filter) {
#if 0
                    bho:
#endif
                        FFTRACE(register ticks t0 = getticks());

                        task = filter->svc(task);

#if defined(TRACE_FASTFLOW)
                        register ticks diff=(getticks()-t0);
                        tickstot +=diff;
                        ticksmin=std::min(ticksmin,diff);
                        ticksmax=std::max(ticksmax,diff);
#endif  

                        if (task == GO_ON) continue;
                        
                        // if the filter returns NULL we exit immediatly
                        if (!task || (task ==(void*)FF_EOS_NOFREEZE)) { 
                            push_eos(true); 
                            ret = task;
                            break; 
                        }
                        if (!task || (task==(void*)FF_EOS)) {
                            push_eos();
                            ret = (void*)FF_EOS;
                            break;
                        }

#if 0
                        // RIVEDERE
                        //-----------------------
                        task = fallback->svc(task);
                        if (task == GO_ON) continue;

                        goto bho;
                        //---------------------------
#endif                        

                    }
                    schedule_task(task);
                }
            } while(1);
        }
        gettimeofday(&wtstop,NULL);
        wttime+=diffmsec(wtstop,wtstart);

        return ret;
    }

    virtual int svc_init() { 
        gettimeofday(&tstart,NULL);

        if (filter && filter->svc_init() <0) return -1;        
        if (fallback && fallback->svc_init()<0) return -1;

        return 0;
    }

    virtual void svc_end() {
        if (filter) filter->svc_end();
        if (fallback) fallback->svc_end();
        gettimeofday(&tstop,NULL);
    }

    int run(bool=false) {
        for(int i=0;i<nworkers;++i) {
            if (workers[i]->run(true)<0) {
                error("LB, spawning worker thread\n");
                return -1;
            }
        }
        if (isfrozen()) 
            for(int i=0;i<nworkers;++i) workers[i]->freeze();
        

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
            error("LB, waiting LB thread\n");
            ret = -1;
        }

        return ret;
    }

    int wait_freezing() {
        int ret=0;
        for(int i=0;i<nworkers;++i)
            if (workers[i]->wait_freezing()<0) {
                error("LB, waiting freezing of worker thread, id = %d\n",workers[i]->get_my_id());
                ret = -1;
            }
        if (ff_thread::wait_freezing()<0) {
            error("LB, waiting LB thread freezing\n");
            ret = -1;
        }
        return ret;
    }

    void stop() {
        for(int i=0;i<nworkers;++i) workers[i]->stop();
        ff_thread::stop();
    }

    void freeze() {
        for(int i=0;i<nworkers;++i) workers[i]->freeze();
        ff_thread::freeze();
    }

    void thaw() {
        for(int i=0;i<nworkers;++i) workers[i]->thaw();
        ff_thread::thaw();
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
        out << "Emitter: "
            << "  work-time (ms): " << wttime    << "\n"
            << "  n. tasks      : " << taskcnt   << "\n"
            << "  svc ticks     : " << tickstot  << " (min= " << (filter?ticksmin:0) << " max= " << ticksmax << ")\n"
            << "  n. push lost  : " << pushwait  << " (ticks=" << lostpushticks << ")" << "\n"
            << "  n. pop lost   : " << popwait   << " (ticks=" << lostpopticks  << ")" << "\n";
    }
#endif

private:
    int                nworkers;
    int                max_nworkers;
    int                nextw;   // out index
    int                nextINw; // in index, used only in master-worker mode
    int                channelid; 
    ff_node         *  filter;
    ff_node        **  workers;
    ff_node         *  fallback;
    FFBUFFER        *  buffer;
    bool               skip1pop;
    bool               master_worker;

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

#endif  /* _FF_LB_HPP_ */
