/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_NODE_HPP_
#define _FF_NODE_HPP_
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

#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <cycle.h>
#include <iostream>
#include <utils.hpp>
#include <buffer.hpp>
#include <ubuffer.hpp>


namespace ff {

#if defined(FF_BOUNDED_BUFFER)
#define FFBUFFER SWSR_Ptr_Buffer
#else  // unbounded buffer
#define FFBUFFER uSWSR_Ptr_Buffer
#endif


enum { FF_EOS=0xffffffff, FF_GO_ON=0x1};

#define GO_ON (void*)ff::FF_GO_ON

#if defined(TRACE_FASTFLOW)
#define FFTRACE(x) x
#else
#define FFTRACE(x)
#endif


class Barrier {
private:
    Barrier():_barrier(0) {
        if (pthread_mutex_init(&bLock,NULL)!=0) {
            error("ERROR: Barrier: pthread_mutex_init fails!\n");
            abort();
        }
        if (pthread_cond_init(&bCond,NULL)!=0) {
            error("ERROR: Barrier: pthread_cond_init fails!\n");
            abort();
        }
    }
public:
    static inline Barrier * instance() {
        static Barrier b;
        return &b;
    }

    inline void barrier(int init = -1) {        
        if (!_barrier && init>0) { _barrier = init; return; }
        
        pthread_mutex_lock(&bLock);
        if (!--_barrier) pthread_cond_broadcast(&bCond);
        else {
            pthread_cond_wait(&bCond, &bLock);
            assert(_barrier==0);
        }
        pthread_mutex_unlock(&bLock);
    }
   
private:
    int _barrier;
    pthread_mutex_t bLock;
    pthread_cond_t  bCond;
};



// forward decl
static void * proxy_thread_routine(void * arg);

class ff_thread {
    friend void * proxy_thread_routine(void *arg);

protected:
    ff_thread():stop(true), // only one shot by default
                spawned(false),
                freezing(false), frozen(false),thawed(false),
                init_error(false) {
        if (pthread_mutex_init(&mutex,NULL)!=0) {
            error("ERROR: ff_thread: pthread_mutex_init fails!\n");
            abort();
        }
        if (pthread_cond_init(&cond,NULL)!=0) {
            error("ERROR: ff_thread: pthread_cond_init fails!\n");
            abort();
        }
        if (pthread_cond_init(&cond_frozen,NULL)!=0) {
            error("ERROR: ff_thread: pthread_cond_init fails!\n");
            abort();
        }
    }
    virtual ~ff_thread() {}

    void thread_routine() {
        Barrier::instance()->barrier();

        do {
            init_error=false;
            if (svc_init()<0) {
                error("ff_thread, svc_init failed, thread exit!!!\n");
                init_error=true;
                break;
            } else  svc(NULL);
            svc_end();
            
            pthread_mutex_lock(&mutex);
            while(freezing) {
                frozen=true; 
                pthread_cond_signal(&cond_frozen);
                pthread_cond_wait(&cond,&mutex);
            }
            thawed=frozen;
            pthread_cond_signal(&cond);
            frozen=false;
            pthread_mutex_unlock(&mutex);

         
        } while(!stop);

        if (init_error && freezing) {
            pthread_mutex_lock(&mutex);
            frozen=true;
            pthread_cond_signal(&cond_frozen);
            pthread_mutex_unlock(&mutex);
        }            
    }

public:
    virtual void* svc(void * task) = 0;
    virtual int   svc_init() { return 0; };
    virtual void  svc_end()  {}

    int spawn() {
        if (spawned) return -1;
        if (pthread_create(&th_handle, NULL, // FIX: attr management !
                           proxy_thread_routine, this) != 0) {
            perror("pthread_create");
            return -1;
        }
        spawned = true;
        return 0;
    }
   
    int wait() {
        stop=true;
        if (isfrozen()) {
            wait_freezing();
            thaw();           
        }
        pthread_join(th_handle, NULL);
        spawned=false;
        return 0;
    }

    int wait_freezing() {
        pthread_mutex_lock(&mutex);
        while(!frozen) pthread_cond_wait(&cond_frozen,&mutex);
        pthread_mutex_unlock(&mutex);        
        return (init_error?-1:0);
    }

    void freeze() {  
        stop=false;
        freezing=true; 
    }
    
    void thaw() {
        pthread_mutex_lock(&mutex);
        freezing=false;
        assert(thawed==false);
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mutex);

        pthread_mutex_lock(&mutex);
        while(!thawed) pthread_cond_wait(&cond, &mutex);
        thawed=false;
        pthread_mutex_unlock(&mutex);
    }

    bool isfrozen() { return freezing;} // tell if freezing has been set

    pthread_t get_handle() const { return th_handle;}

private:
    bool            stop;
    bool            spawned;
    bool            freezing;
    bool            frozen;
    bool            thawed;
    bool            init_error;
    pthread_t       th_handle;
    pthread_attr_t  attr; // TODO
    pthread_mutex_t mutex; 
    pthread_cond_t  cond;
    pthread_cond_t  cond_frozen;
};
    
static void * proxy_thread_routine(void * arg) {
    ff_thread & obj = *(ff_thread *)arg;
    obj.thread_routine();
    pthread_exit(NULL);
}



class ff_node {
private:

    template <typename lb_t, typename gt_t> 
    friend class ff_farm;
    friend class ff_pipeline;
    friend class ff_loadbalancer;


    void set_id(int id) { myid = id;}
    bool push(void * ptr) { return out->push(ptr); }
    bool pop(void ** ptr) { return ((in->pop(ptr)>0)?true:false);} 
    bool skipfirstpop() const { return skip1pop; }
    
    virtual int create_input_buffer(int nentries) {
        if (in) return -1;
        in = new FFBUFFER(nentries);        
        if (!in) return -1;
        myinbuffer=true;
        return (in->init()?0:-1);
    }
    
    virtual int create_output_buffer(int nentries) {
        if (out) return -1;
        out = new FFBUFFER(nentries);        
        if (!out) return -1;
        myoutbuffer=true;
        return (out->init()?0:-1);
    }

    virtual int set_output_buffer(FFBUFFER * const o) {
        if (myoutbuffer) return -1;
        out = o;
        return 0;
    }

    virtual int set_input_buffer(FFBUFFER * const i) {
        if (myinbuffer) return -1;
        in = i;
        return 0;
    }

    virtual int   run(bool=false) { 
        thread = new thWorker(this);
        if (!thread) return -1;
        return thread->run();
    }

    virtual int  wait() { 
        if (!thread) return -1;
        return thread->wait(); 
    }
    virtual int  wait_freezing() { 
        if (!thread) return -1;
        return thread->wait_freezing(); 
    }
    virtual void freeze() { 
        if (!thread) return; 
        thread->freeze(); 
    }
    virtual void thaw() { 
        if (!thread) return; 
        thread->thaw();
    }
    virtual bool isfrozen() { 
        if (!thread) 
            return false;
        return thread->isfrozen();
    }
    virtual int  cardinality() const { return 1;}

    virtual double ffTime() {
        return diffmsec(tstop,tstart);
    }

    virtual double wffTime() {
        return diffmsec(wtstop,wtstart);
    }

public:
    // If svc returns a NULL value then End-Of-Stream (EOS) is produced
    // on the output channel.
    virtual void* svc(void * task) = 0;
    virtual int   svc_init() { return 0; }
    virtual void  svc_end() {}
    virtual int   get_my_id() const { return myid; };
    virtual bool  put(void * ptr) { return in->push(ptr);}
    virtual bool  get(void **ptr) { return ((out->pop(ptr)>0)?true:false);}
    virtual FFBUFFER * const get_in_buffer() const { return in;}
    virtual FFBUFFER * const get_out_buffer() const { return out;}

    virtual const struct timeval & getstarttime() const { return tstart;}
    virtual const struct timeval & getstoptime()  const { return tstop;}
    virtual const struct timeval & getwstartime() const { return wtstart;}
    virtual const struct timeval & getwstoptime() const { return wtstop;}
#if defined(TRACE_FASTFLOW)    
    virtual void ffStats(std::ostream & out) { 
        out << "ID: " << get_my_id()
            << "  work-time (ms): " << wffTime() << "\n"
            << "  n. tasks      : " << taskcnt   << "\n"
            << "  svc ticks     : " << tickstot  << " (min= " << ticksmin << " max= " << ticksmax << ")\n"
            << "  n. push lost  : " << pushwait  << " (ticks=" << lostpushticks << ")" << "\n"
            << "  n. pop lost   : " << popwait   << " (ticks=" << lostpopticks  << ")" << "\n";
    }
#endif

protected:
    // protected constructor
    ff_node():in(0),out(0),myid(-1),
              myoutbuffer(false),myinbuffer(false),
              skip1pop(false), thread(NULL),callback(NULL) {
        tstart.tv_sec=0;  tstart.tv_usec=0;
        tstop.tv_sec=0;   tstop.tv_usec=0;
        wtstart.tv_sec=0; wtstart.tv_usec=0;
        wtstop.tv_sec=0;  wtstop.tv_usec=0;
        FFTRACE(taskcnt=0;lostpushticks=0;pushwait=0;lostpopticks=0;popwait=0;ticksmin=(ticks)-1;ticksmax=0;tickstot=0);
    };
    
    virtual  ~ff_node() {
        if (in && myinbuffer)  delete in;
        if (out && myoutbuffer) delete out;
        if (thread) delete thread;
    };
    
    virtual bool ff_send_out(void * task, 
                             unsigned int retry=((unsigned int)-1),
                             unsigned int ticks=thWorker::TICKS2WAIT) {

        if (callback) return  callback(task,retry,ticks,callback_arg);

        for(unsigned int i=0;i<retry;++i) {
            if (push(task)) return true;
            ticks_wait(ticks);
        }     
        return false;
    }

private:

    void registerCallback(bool (*cb)(void *,unsigned int,unsigned int,void *), void * arg) {
        callback=cb;
        callback_arg=arg;
    }

    class thWorker: public ff_thread {
    public:
        enum {TICKS2WAIT=1000};

        thWorker(ff_node * const filter):filter(filter) {}
        
        bool push(void * task) {
            //register int cnt = 0;
            while (! filter->push(task)) {
                // if (ch->thxcore>1) {
            // if (++cnt>PUSH_POP_CNT) { sched_yield(); cnt=0;}
            //    else ticks_wait(TICKS2WAIT);
            //} else 

                FFTRACE(filter->lostpushticks+=TICKS2WAIT; ++filter->pushwait);
                ticks_wait(TICKS2WAIT);
            }     
            return true;
        }
        
        bool pop(void ** task) {
            //register int cnt = 0;       
            while (! filter->pop(task)) {
                //    if (ch->thxcore>1) {
                //if (++cnt>PUSH_POP_CNT) { sched_yield(); cnt=0;}
                //else ticks_wait(TICKS2WAIT);
                //} else 

                FFTRACE(filter->lostpopticks+=TICKS2WAIT; ++filter->popwait);
                ticks_wait(TICKS2WAIT);
            } 
            return true;
        }
        
        bool put(void * ptr) { return filter->put(ptr);}
        bool get(void **ptr) { return filter->get(ptr);}
        
        void* svc(void * ) {
            void * task = NULL;
            bool inpresent  = (filter->get_in_buffer() != NULL);
            bool outpresent = (filter->get_out_buffer() != NULL);
            bool skipfirstpop = filter->skipfirstpop(); 
            bool exit=false;

            gettimeofday(&filter->wtstart,NULL);
            do {
                if (inpresent) {
                    if (!skipfirstpop) pop(&task); 
                    else skipfirstpop=false;
                    if (task == (void*)FF_EOS) { 
                        if (outpresent)  push(task); 
                        break;
                    }
                }
                FFTRACE(++filter->taskcnt);
                FFTRACE(register ticks t0 = getticks());

                void * result = filter->svc(task);

#if defined(TRACE_FASTFLOW)
                register ticks diff=(getticks()-t0);
                filter->tickstot +=diff;
                filter->ticksmin=std::min(filter->ticksmin,diff);
                filter->ticksmax=std::max(filter->ticksmax,diff);
#endif                

                if (!result) {
                    // NOTE: The EOS is gonna be produced in the output queue
                    // and the thread is exiting even if there might be some tasks
                    // in the input queue !!!
                    result = (void *)FF_EOS; 
                    exit=true;
                }
                if (outpresent && (result != GO_ON)) push(result);
            } while(!exit);
            
            gettimeofday(&filter->wtstop,NULL);
            return NULL;
        }
        
        int svc_init() { 
            gettimeofday(&filter->tstart,NULL);
            return filter->svc_init(); 
        }
        
        virtual void svc_end() {
            filter->svc_end();
            gettimeofday(&filter->tstop,NULL);
        }
        
        int run() { return ff_thread::spawn(); }
        int wait() { return ff_thread::wait();}
        int wait_freezing() { return ff_thread::wait_freezing();}
        void freeze() { ff_thread::freeze();}
        bool isfrozen() { return ff_thread::isfrozen();}

        int get_my_id() const { return filter->get_my_id(); };
        
    protected:    
        ff_node * const filter;
    };

private:
    FFBUFFER         * in;
    FFBUFFER         * out;
    int               myid;
    bool              myoutbuffer;
    bool              myinbuffer;
    bool              skip1pop;
    thWorker        * thread;
    bool (*callback)(void *,unsigned int,unsigned int, void *);
    void * callback_arg;

    struct timeval tstart;
    struct timeval tstop;
    struct timeval wtstart;
    struct timeval wtstop;

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

#endif /* _FF_NODE_HPP_ */
