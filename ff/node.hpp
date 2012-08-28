/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file node.hpp
 *  \brief This file contains the definition of the \p ff_node class, which acts as the basic 
 *  structure of each skeleton. Other supplementary classes are defined here. *
 */

#ifndef _FF_NODE_HPP_
#define _FF_NODE_HPP_
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

#include <stdlib.h>
//#include <sys/time.h>
//#include <pthread.h>
#include <iostream>
#include "ff/platforms/platform.h"
#include <ff/cycle.h>
#include <ff/utils.hpp>
#include <ff/buffer.hpp>
#include <ff/ubuffer.hpp>


namespace ff {




/*
 * NOTE: by default FF_BOUNDED_BUFFER is not defined
 */
#if defined(FF_BOUNDED_BUFFER)
#define FFBUFFER SWSR_Ptr_Buffer
#else  // unbounded buffer
#define FFBUFFER uSWSR_Ptr_Buffer
#endif


enum { FF_EOS=0xffffffff, FF_EOS_NOFREEZE=(FF_EOS-0x1) , FF_GO_ON=(FF_EOS-0x2)};

#define GO_ON         (void*)ff::FF_GO_ON
#define EOS_NOFREEZE  (void*)ff::FF_EOS_NOFREEZE
#define EOS           (void*)ff::FF_EOS

#if defined(TRACE_FASTFLOW)
#define FFTRACE(x) x
#else
#define FFTRACE(x)
#endif

/*!
 *  \ingroup runtime
 *
 *  @{
 */

/*!
 *  \class Barrier
 *
 *  \brief A class representing a Barrier, i.e. a \a Mutex
 *
 *  <em>SHALL WE INCLUDE THIS CLASS IN THE DOCUMENTATION?</em> \n
 *  This class provides the methods necessary to implement a low-level \p pthread mutex.
 */
class Barrier {
public:
    static inline Barrier * instance() {
        static Barrier b;
        return &b;
    }

    /*!
     *  Default Constructor.
     *
     *  Checks whether the mutex variable(s) and the conditional variable(s) can be propetly 
     *  initialised
     */
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
    
    /** Performs the barrier and waits on the condition variable */
    inline void doBarrier(int init = -1) {        
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
    int _barrier;           /// 
    pthread_mutex_t bLock;  /// Mutex variable
    pthread_cond_t  bCond;  /// Condition variable
};

/*!
 * @}
 */



// forward decl
static void * proxy_thread_routine(void * arg);

/*!
 *  \ingroup runtime
 *
 *  @{
 */

/*!
 *  \class ff_thread
 *
 *  \brief This class contains methods to describe a ff_thread. 
 *
 *  This class performs all the low-level operations need to manage threads communication and 
 *  synchronisation. It is responsible for threads creation and destruction, suspension, freezing and 
 *  thawing.
 *  
 */
class ff_thread {
    friend void * proxy_thread_routine(void *arg);

protected:
    /*! Constructor 
     *  @param[in] barrier Takes a Barrier object as input paramenter.
     */
    ff_thread(Barrier * barrier=NULL):
        barrier(barrier),
        stp(true), // only one shot by default
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

    /// Default destructor
    virtual ~ff_thread() {
        // MarcoA 27/04/12: Moved to wait
        /*
        if (pthread_attr_destroy(&attr)) {
            error("ERROR: ~ff_thread: pthread_attr_destroy fails!");
        }
        */
    }
    
    /** Control thread's life cycle  */
    void thread_routine() {
        //Barrier::instance()->barrier();
        if (barrier) barrier->doBarrier();
        void * ret;
        do {
            init_error=false;
            if (svc_init()<0) {
                error("ff_thread, svc_init failed, thread exit!!!\n");
                init_error=true;
                break;
            } else  ret = svc(NULL);
            svc_end();
            
            if (disable_cancelability()) {
                error("ff_thread, thread_routine, could not change thread cancelability");
                return;
            }

            pthread_mutex_lock(&mutex);
            if (ret != (void*)FF_EOS_NOFREEZE) {
                while(freezing) {
                    frozen=true; 
                    pthread_cond_signal(&cond_frozen);
                    pthread_cond_wait(&cond,&mutex);
                }
            }

            thawed=frozen;
            pthread_cond_signal(&cond);
            frozen=false;            
            pthread_mutex_unlock(&mutex);

            if (enable_cancelability()) {
                error("ff_thread, thread_routine, could not change thread cancelability");
                return;
            }

         
        } while(!stp);

        if (init_error && freezing) {
            pthread_mutex_lock(&mutex);
            frozen=true;
            pthread_cond_signal(&cond_frozen);
            pthread_mutex_unlock(&mutex);
        } 
    }

    int disable_cancelability()
    {
        if (pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old_cancelstate)) {
            perror("pthread_setcanceltype");
            return -1;
        }
        return 0;
    }

    int enable_cancelability()
    {
        if (pthread_setcancelstate(old_cancelstate, 0)) {
            perror("pthread_setcanceltype");
            return -1;
        }
        return 0;
    }
    
public:
    virtual void* svc(void * task) = 0;
    virtual int   svc_init() { return 0; };
    virtual void  svc_end()  {}

    void set_barrier(Barrier * const b) { barrier=b;}

    /// Creates a new thread
    int spawn() {
        if (spawned) return -1;
        
        if (pthread_attr_init(&attr)) {
                perror("pthread_attr_init");
                return -1;
        }
        if (pthread_create(&th_handle, &attr,
                           proxy_thread_routine, this) != 0) {
            perror("pthread_create");
            return -1;
        }
        spawned = true;
        return 0;
    }
   
    /// Waits thread termination
    int wait() {
        stp=true;
        if (isfrozen()) {
            wait_freezing();
            thaw();           
        }
        if (spawned)
            pthread_join(th_handle, NULL);
        if (pthread_attr_destroy(&attr)) {
            error("ERROR: ff_thread.wait: pthread_attr_destroy fails!");
        }
        spawned=false;
        return 0;
    }

    /// Waits thread freezing
    int wait_freezing() {
        pthread_mutex_lock(&mutex);
        while(!frozen) pthread_cond_wait(&cond_frozen,&mutex);        
        pthread_mutex_unlock(&mutex);        
        return (init_error?-1:0);
    }

    /// Force the thread to stop at next EOS or next thawing
    void stop() { stp = true; };

    /// Force the thread to freeze himself
    void freeze() {  
        stp=false;
        freezing=true; 
    }
    
    /// If the thread is frozen, then thaw it
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

    /// Tell if the thread is frozen
    bool isfrozen() { return freezing;} 

    pthread_t get_handle() const { return th_handle;}

private:
    Barrier      *  barrier;            /// A \p Barrier object
    bool            stp;
    bool            spawned;
    bool            freezing;
    bool            frozen;
    bool            thawed;
    bool            init_error;
    pthread_t       th_handle;
    pthread_attr_t  attr;
    pthread_mutex_t mutex; 
    pthread_cond_t  cond;
    pthread_cond_t  cond_frozen;
    int             old_cancelstate;
};
    
static void * proxy_thread_routine(void * arg) {
    ff_thread & obj = *(ff_thread *)arg;
    obj.thread_routine();
    pthread_exit(NULL);
    return NULL;
}

/*!
 *  @}
 */

/*!
 *  \ingroup low_level
 *
 *  @{
 */

/*!
 *  \class ff_node
 *
 *  \brief This class describes the \p ff_node, the basic building block of every skeleton.
 *
 *  This class desribes the Node object, the base class of every skeleton. \p ff_node defines 3 basic 
 *  methods, two optional - \p svc_init and \p svc_end - and one mandatory - \p svc (pure virtual 
 *  method). The \p svc_init method is called once at node initialization, while the \p svn_end method 
 *  is called once when the end-of-stream (EOS) is received in input or when the \p svc method returns 
 *  \p NULL. The \p svc method is called each time an input task is ready to be processed.
 */
class ff_node {
private:

    template <typename lb_t, typename gt_t> 
    friend class ff_farm;
    friend class ff_pipeline;
    friend class ff_loadbalancer;
    friend class ff_gatherer;

protected:
    void set_id(int id) { myid = id;}
    virtual inline bool push(void * ptr) { return out->push(ptr); }
    virtual inline bool pop(void ** ptr) { return in->pop(ptr); } 
    virtual inline void skipfirstpop(bool sk)   { skip1pop=sk;}
    bool skipfirstpop() const { return skip1pop; }
    
    virtual int create_input_buffer(int nentries, bool fixedsize=true) {
        if (in) return -1;
        in = new FFBUFFER(nentries,fixedsize);        
        if (!in) return -1;
        myinbuffer=true;
        return (in->init()?0:-1);
    }
    
    virtual int create_output_buffer(int nentries, bool fixedsize=false) {
        if (out) return -1;
        out = new FFBUFFER(nentries,fixedsize);        
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

    virtual int freeze_and_run(bool=false) {
        thread = new thWorker(this);
        if (!thread) return -1;
        freeze();
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
    virtual void stop() {
        if (!thread) return; 
        thread->stop(); 
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
    virtual int  cardinality(Barrier * const b) { 
        barrier = b;
        return 1;
    }

    virtual double ffTime() {
        return diffmsec(tstop,tstart);
    }

    virtual double wffTime() {
        return diffmsec(wtstop,wtstart);
    }

public:
	enum {TICKS2WAIT=1000};
    // If svc returns a NULL value then End-Of-Stream (EOS) is produced
    // on the output channel.
    virtual void* svc(void * task) = 0;
    virtual int   svc_init() { return 0; }
    virtual void  svc_end() {}
    virtual int   get_my_id() const { return myid; };
#if defined(TEST_QUEUE_SPIN_LOCK)
    virtual bool  put(void * ptr) { 
        spin_lock(lock);
        bool r= in->push(ptr);
        spin_unlock(lock);
        return r;
    }
    virtual bool  get(void **ptr) { 
        spin_lock(lock);
        register bool r = out->pop(ptr);
        spin_unlock(lock);
        return r;
    }
#else
    virtual bool  put(void * ptr) { return in->push(ptr);}
    virtual bool  get(void **ptr) { return out->pop(ptr);}
#endif
    virtual FFBUFFER * const get_in_buffer() const { return in;}
    virtual FFBUFFER * const get_out_buffer() const { return out;}

    virtual const struct timeval & getstarttime() const { return tstart;}
    virtual const struct timeval & getstoptime()  const { return tstop;}
    virtual const struct timeval & getwstartime() const { return wtstart;}
    virtual const struct timeval & getwstoptime() const { return wtstop;}
#if defined(TRACE_FASTFLOW)    
    virtual void ffStats(std::ostream & out) { 
        out << "ID: " << get_my_id()
            << "  work-time (ms): " << wttime    << "\n"
            << "  n. tasks      : " << taskcnt   << "\n"
            << "  svc ticks     : " << tickstot  << " (min= " << ticksmin << " max= " << ticksmax << ")\n"
            << "  n. push lost  : " << pushwait  << " (ticks=" << lostpushticks << ")" << "\n"
            << "  n. pop lost   : " << popwait   << " (ticks=" << lostpopticks  << ")" << "\n";
    }
#endif

protected:
    /// Protected constructor
    ff_node():in(0),out(0),myid(-1),
              myoutbuffer(false),myinbuffer(false),
              skip1pop(false), thread(NULL),callback(NULL),barrier(NULL) {
        time_setzero(tstart);time_setzero(tstop);
        time_setzero(wtstart);time_setzero(wtstop);
        wttime=0;
        FFTRACE(taskcnt=0;lostpushticks=0;pushwait=0;lostpopticks=0;popwait=0;ticksmin=(ticks)-1;ticksmax=0;tickstot=0);
#if defined(TEST_QUEUE_SPIN_LOCK)
        init_unlocked(lock);
#endif
    };
    
    /** 
     *  Destructor 
     *
     *  Deletes input and output buffers and the working threads
     */
    virtual  ~ff_node() {
        if (in && myinbuffer)  delete in;
        if (out && myoutbuffer) delete out;
        if (thread) delete thread;
    };
    
    /** Allows to queue tasks without returning from the \p svc method */
    virtual bool ff_send_out(void * task, 
                             unsigned int retry=((unsigned int)-1),
                             unsigned int ticks=(TICKS2WAIT)) { 
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
    
    /*!
     *  @}
     */

    /* Inner class that wraps ff_thread functions and adds \p push and \p pop methods */
    class thWorker: public ff_thread {
    public:
        thWorker(ff_node * const filter):
            ff_thread(filter->barrier),filter(filter) {}
        
        inline bool push(void * task) {
            //register int cnt = 0;
            /* NOTE: filter->push and not buffer->push because of the filter can be a dnode
             */
            while (! filter->push(task)) { 
                // if (ch->thxcore>1) {
                // if (++cnt>PUSH_POP_CNT) { sched_yield(); cnt=0;}
                //    else ticks_wait(TICKS2WAIT);
                //} else 

                FFTRACE(filter->lostpushticks+=ff_node::TICKS2WAIT; ++filter->pushwait);
                ticks_wait(ff_node::TICKS2WAIT);
            }     
            return true;
        }
        
        inline bool pop(void ** task) {
            //register int cnt = 0;       
            /* NOTE: filter->push and not buffer->push because of the filter can be a dnode
             */
            while (! filter->pop(task)) {
                //if (ch->thxcore>1) {
                //if (++cnt>PUSH_POP_CNT) { sched_yield(); cnt=0;}
                //else ticks_wait(TICKS2WAIT);
                //} else 

                FFTRACE(filter->lostpopticks+=ff_node::TICKS2WAIT; ++filter->popwait);
                ticks_wait(ff_node::TICKS2WAIT);
            } 
            return true;
        }
        
        inline bool put(void * ptr) { return filter->put(ptr);}
        inline bool get(void **ptr) { return filter->get(ptr);}
        
        void* svc(void * ) {
            void * task = NULL;
            void * ret = (void*)FF_EOS;
            bool inpresent  = (filter->get_in_buffer() != NULL);
            bool outpresent = (filter->get_out_buffer() != NULL);
            bool skipfirstpop = filter->skipfirstpop(); 
            bool exit=false;            

            gettimeofday(&filter->wtstart,NULL);
            do {
                if (inpresent) {
                    if (!skipfirstpop) pop(&task); 
                    else skipfirstpop=false;
                    if ((task == (void*)FF_EOS) || 
                        (task == (void*)FF_EOS_NOFREEZE)) {
                        ret = task;
                        if (outpresent)  push(task); 
                        break;
                    }
                }
                FFTRACE(++filter->taskcnt);
                FFTRACE(register ticks t0 = getticks());

                ret = filter->svc(task);

#if defined(TRACE_FASTFLOW)
                register ticks diff=(getticks()-t0);
                filter->tickstot +=diff;
                filter->ticksmin=(std::min)(filter->ticksmin,diff); // (std::min) for win portability)
                filter->ticksmax=(std::max)(filter->ticksmax,diff);
#endif                
                if (!ret || (ret == (void*)FF_EOS) || (ret == (void*)FF_EOS_NOFREEZE)) {
                    // NOTE: The EOS is gonna be produced in the output queue
                    // and the thread exits even if there might be some tasks
                    // in the input queue !!!
                    if (!ret) ret = (void *)FF_EOS;
                    exit=true;
                }
                if (outpresent && (ret != GO_ON)) push(ret);
            } while(!exit);
            
            gettimeofday(&filter->wtstop,NULL);
            filter->wttime+=diffmsec(filter->wtstop,filter->wtstart);

            return ret;
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
    
    
/*!
 *  \ingroup low_level
 *
 *  @{
 */

private:
    FFBUFFER        * in;           /// Can be either a \p SWSR_Ptr_Buffer or an \p uSWSR_Ptr_Buffer
    FFBUFFER        * out;          /// Can be either a \p SWSR_Ptr_Buffer or an \p uSWSR_Ptr_Buffer
    int               myid;
    bool              myoutbuffer;
    bool              myinbuffer;
    bool              skip1pop;
    thWorker        * thread;       /// A \p thWorker object, which extends the \p ff_thread class 
    bool (*callback)(void *,unsigned int,unsigned int, void *);
    void            * callback_arg;
    Barrier         * barrier;      /// A \p Barrier object

    struct timeval tstart;
    struct timeval tstop;
    struct timeval wtstart;
    struct timeval wtstop;
    double wttime;

#if defined(TEST_QUEUE_SPIN_LOCK)
    lock_t lock;
#endif

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

/*!
 *  @}
 */


} // namespace ff

#endif /* _FF_NODE_HPP_ */
