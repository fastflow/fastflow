/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file node.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow ff_node 
 *
 * @detail FastFlow basic contanier for a shared-memory parallel activity 
 *
 */

#ifndef FF_NODE_HPP
#define FF_NODE_HPP

/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

#include <stdlib.h>
#include <iostream>
#include <ff/platforms/platform.h>
#include <ff/cycle.h>
#include <ff/utils.hpp>
#include <ff/buffer.hpp>
#include <ff/ubuffer.hpp>
#include <ff/mapper.hpp>
#include <ff/config.hpp>
#include <ff/svector.hpp>
#include <ff/barrier.hpp>

static void *GO_ON        = (void*)ff::FF_GO_ON;
static void *GO_OUT       = (void*)ff::FF_GO_OUT;
static void *EOS_NOFREEZE = (void*)ff::FF_EOS_NOFREEZE;
static void *EOS          = (void*)ff::FF_EOS;


namespace ff {

struct fftree;


// TODO: Should be rewritten in terms of mapping_utils.hpp 
#if defined(HAVE_PTHREAD_SETAFFINITY_NP) && !defined(NO_DEFAULT_MAPPING)

    /*
     *
     * \brief Initialize thread affinity 
     * It initializes thread affinity i.e. which cpu the thread should be
     * assigned.
     *
     * \param attr is the pthread attribute
     * \param cpuID is the identifier the core
     * \return -2  if error, the cpu identifier if successful
     */
static inline int init_thread_affinity(pthread_attr_t*attr, int cpuId) {
    cpu_set_t cpuset;    
    CPU_ZERO(&cpuset);

    int id;
    if (cpuId<0) {
        id = threadMapper::instance()->getCoreId();
        CPU_SET (id, &cpuset);
    } else  {
        id = cpuId;
        CPU_SET (cpuId, &cpuset);
    }

    if (pthread_attr_setaffinity_np (attr, sizeof(cpuset), &cpuset)<0) {
        perror("pthread_attr_setaffinity_np");
        return -2;
    }
    return id;    
}
#elif !defined(HAVE_PTHREAD_SETAFFINITY_NP) && !defined(NO_DEFAULT_MAPPING)

/*
 * \brief Initializes thread affinity
 *
 * It initializes thread affinity i.e. it defines to which core ths thread
 * should be assigned.
 *
 * \return always return -1 because no thread mapping is done
 */
static inline int init_thread_affinity(pthread_attr_t*,int) {
    // Just ensure that the threadMapper constructor is called.
    threadMapper::instance();
    return -1;
}
#else
/*
 * \brief Initializes thread affinity
 *
 * It initializes thread affinity i.e. it defines to which core ths thread
 * should be assigned.
 *
 * \return always return -1 because no thread mapping is done
 */
static inline int init_thread_affinity(pthread_attr_t*,int) {
    // do nothing
    return -1;
}
#endif /* HAVE_PTHREAD_SETAFFINITY_NP */


// forward decl
/*
 * \brief Proxy thread routine
 *
 */
static void * proxy_thread_routine(void * arg);

/*!
 *  \class ff_thread
 *  \ingroup buiding_blocks
 *
 *  \brief thread container for (leaves) ff_node
 *
 * It defines FastFlow's threading abstraction to run ff_node in parallel
 * in the shared-memory runtime
 *
 * \note Should not be used directly, it is called by ff_node
 */
class ff_thread {

    friend void * proxy_thread_routine(void *arg);

protected:
    ff_thread(BARRIER_T * barrier=NULL):
    	fftree_ptr(NULL),
        tid((unsigned)-1),barrier(barrier),
        stp(true), // only one shot by default
        spawned(false),
        freezing(0), frozen(false), //thawed(false),
        init_error(false) {
        
        /* Attr is NULL, default mutex attributes are used. Upon successful
         * initialization, the state of the mutex becomes initialized and
         * unlocked. 
         * */
        if (pthread_mutex_init(&mutex,NULL)!=0) {
            error("FATAL ERROR: ff_thread: pthread_mutex_init fails!\n");
            abort();
        }
        if (pthread_cond_init(&cond,NULL)!=0) {
            error("FATAL ERROR: ff_thread: pthread_cond_init fails!\n");
            abort();
        }
        if (pthread_cond_init(&cond_frozen,NULL)!=0) {
            error("FATAL ERROR: ff_thread: pthread_cond_init fails!\n");
            abort();
        }
    }

    virtual ~ff_thread() {}
    
    void thread_routine() {
        if (barrier) barrier->doBarrier(tid);
        void * ret;
        do {
            init_error=false;
            if (svc_init()<0) {
                error("ff_thread, svc_init failed, thread exit!!!\n");
                init_error=true;
                break;
            } else  {
                ret = svc(NULL);
                svc_releaseOCL();
            }
            svc_end();
            
            if (disable_cancelability()) {
                error("ff_thread, thread_routine, could not change thread cancelability");
                return;
            }

            // acquire lock. While freezing is true,
            // freeze and wait. 
            pthread_mutex_lock(&mutex);
            if (ret != EOS_NOFREEZE && !stp) {
                if (freezing == 0 && ret == (void*)FF_EOS) stp = true;
                while(freezing==1) { // NOTE: freezing can change to 2
                    frozen=true; 
                    pthread_cond_signal(&cond_frozen);
                    pthread_cond_wait(&cond,&mutex);
                }
            }
            
            //thawed=true;
            //pthread_cond_signal(&cond);
            //frozen=false; 
            if (freezing != 0) freezing = 1; // freeze again next time 
            pthread_mutex_unlock(&mutex);

            if (enable_cancelability()) {
                error("ff_thread, thread_routine, could not change thread cancelability");
                return;
            }
        } while(!stp);
        
        if (freezing) {
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

    /*
     * \brief The OpenCL initialisation
     *
     */
    virtual void  svc_createOCL()  {}

    /*
     * \brief OpenCL finsalisation
     *
     */
    virtual void  svc_releaseOCL() {}

    void set_barrier(BARRIER_T * const b) { barrier=b;}

    int spawn(int cpuId=-1) {
        if (spawned) return -1;

        if (pthread_attr_init(&attr)) {
                perror("pthread_attr_init: pthread can not be created.");
                return -1;
        }

        int CPUId = init_thread_affinity(&attr, cpuId);
        if (CPUId==-2) return -2;
        if (barrier) tid= (unsigned) barrier->getCounter();
        if (pthread_create(&th_handle, &attr,
                           proxy_thread_routine, this) != 0) {
            perror("pthread_create: pthread creation failed.");
            return -2;
        }
        if (barrier) barrier->incCounter();
        spawned = true;
        return CPUId;
    }
   
  
    int wait() {
        int r=0;
        stp=true;
        if (isfrozen()) {
            wait_freezing();
            thaw();
        }
        if (spawned) {
            pthread_join(th_handle, NULL);
            if (barrier) barrier->decCounter();
        }
        if (pthread_attr_destroy(&attr)) {
            error("ERROR: ff_thread.wait: pthread_attr_destroy fails!");
            r=-1;
        }
        spawned=false;
        return r;
    }

    inline int wait_freezing() {
        pthread_mutex_lock(&mutex);
        while(!frozen) pthread_cond_wait(&cond_frozen,&mutex);
        pthread_mutex_unlock(&mutex);
        return (init_error?-1:0);
    }

    inline void stop() { stp = true; };

    inline void freeze() {  
        stp=false;
        freezing = 1;
    }
    
    inline void thaw(bool _freeze=false) {
        pthread_mutex_lock(&mutex);
        // if this function is called even if the thread is not 
        // in frozen state, then freezing has to be set to 1 and not 2
        //if (_freeze) freezing= (frozen?2:1); // next time freeze again the thread
        // October 2014, changed the above policy.
        // If thaw is called and the thread is not in the frozen stage, 
        // then the thread won't fall to sleep at the next freezing point

        if (_freeze) freezing = 2; // next time freeze again the thread
        else freezing=0;
        //assert(thawed==false);
        frozen=false; 
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mutex);

        //pthread_mutex_lock(&mutex);
        //while(!thawed) pthread_cond_wait(&cond, &mutex);
        //thawed=false;
        //pthread_mutex_unlock(&mutex);
    }

    inline bool isfrozen() const { return freezing>0;} 

    inline bool iswaiting() const { return frozen;}

    pthread_t get_handle() const { return th_handle;}

    inline int getTid() const { return tid; }

    //fftree stuff
    fftree *fftree_ptr;

protected:
    unsigned        tid;
private:
    BARRIER_T    *  barrier;            /// A \p Barrier object
    bool            stp;
    bool            spawned;
    int             freezing;   // changed from bool to int
    bool            frozen;
    //bool            thawed;
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
 *  \class ff_node
 *  \ingroup building_blocks
 *
 *  \brief The FastFlow abstract contanier for a parallel activity (actor).
 *
 * Implements \p ff_node, i.e. the general container for a parallel
 * activity. From the orchestration viewpoint, the process model to
 * be employed is a CSP/Actor hybrid model where activities (\p
 * ff_nodes) are named and the data paths between processes are
 * clearly identified. \p ff_nodes synchronise each another via
 * abstract units of SPSC communications and synchronisation (namely
 * 1:1 channels), which models data dependency between two
 * \p ff_nodes.  It is used to encapsulate
 * sequential portions of code implementing functions. 
 *
 * \p In a multicore, a ff_node is implemented as non-blocking thread. 
 * It is not and should
 * not be confused with a task. Typically a \p ff_node uses the 100% of one CPU
 * context (i.e. one core, either physical or HT, if any). Overall, the number of
 * ff_nodes running should not exceed the number of logical cores of the platform.
 * 
 * \p A ff_node behaves as a loop that gets an input (i.e. the parameter of \p svc 
 * method) and produces one or more outputs (i.e. return parameter of \p svc method 
 * or parameter of the \p ff_send_out method that can be called in the \p svc method). 
 * The loop complete on the output of the special value "end-of_stream" (EOS). 
 * The EOS is propagated across channels to the next \p ff_node.  
 * 
 * Key methods are: \p svc_init, \p svc_end (optional), and \p svc (pure virtual, 
 * mandatory). The \p svc_init method is called once at node initialization,
 * while the \p svn_end method is called after a EOS task has been returned. 
 *
 *  This class is defined in \ref node.hpp
 */

class ff_node {
private:

    template <typename lb_t, typename gt_t> 
    friend class ff_farm;
    friend class ff_pipeline;
    friend class ff_map;
    friend class ff_loadbalancer;
    friend class ff_gatherer;
    friend class ff_ofarm;

protected:
    
    void set_id(int id) { myid = id;}
    
#if defined(TEST_QUEUE_LOCK)
    virtual inline bool push(void * ptr) { 
        pthread_mutex_lock(&mutex_out);
        while(!out->push(ptr)) 
            pthread_cond_wait(&queue_out_task_out,&mutex_out);
        pthread_cond_signal(&queue_out_task_in);
        pthread_mutex_unlock(&mutex_out);
        return true;
    }
    virtual inline bool pop(void ** ptr) { 
        if (!in_active) return false; // it does not want to receive data
        
        pthread_mutex_lock(&mutex_in);
        while(!in->pop(ptr))
            pthread_cond_wait(&queue_in_task_in,&mutex_in);
        pthread_cond_signal(&queue_in_task_out);
        pthread_mutex_unlock(&mutex_in);
        return true;
    }
#else
    virtual inline bool push(void * ptr) { return out->push(ptr); }
    virtual inline bool pop(void ** ptr) { 
        if (!in_active) return false; // it does not want to receive data
        return in->pop(ptr);
    }
#endif

    /**
     * \brief Set the ff_node to start with no input task
     *
     * Setting it to true let the \p ff_node execute the \p svc method spontaneusly 
     * before receiving a task on the input channel. \p skipfirstpop makes it possible
     * to define a "producer" node that starts the network.
     *
     * \param sk \p true start spontaneously (*task will be NULL)
     *
     */
    virtual inline void skipfirstpop(bool sk)   { skip1pop=sk;}

    /** 
     * \brief Gets the status of spontaneous start
     * 
     * If \p true the \p ff_node execute the \p svc method spontaneusly
     * before receiving a task on the input channel. \p skipfirstpop makes it possible
     * to define a "producer" node that produce the stream.
     * 
     * \return \p true if skip-the-first-element mode is set, \p false otherwise
     * 
     * Example: \ref l1_ff_nodes_graph.cpp
     */
    bool skipfirstpop() const { return skip1pop; }
    
    /** 
     * \brief Creates the input channel 
     *
     *  \param nentries: the number of elements of the buffer
     *  \param fixedsize flag to decide whether the buffer is bound or unbound.
     *  Default is \p true.
     *
     *  \return 0 if successful, -1 otherwise
     */
    virtual int create_input_buffer(int nentries, bool fixedsize=true) {
        if (in) return -1;
        // MA: here probably better to use p = posix_memalign to 64 bits; new (p) FFBUFFER
		in = new FFBUFFER(nentries,fixedsize);        
        if (!in) return -1;
        myinbuffer=true;
        return (in->init()?0:-1);
    }
    
    /** 
     *  \brief Creates the output channel
     *
     *  \param nentries: the number of elements of the buffer
     *  \param fixedsize flag to decide whether the buffer is bound or unbound.
     *  Default is \p true.
     *
     *  \return 0 if successful, -1 otherwise
     */
    virtual int create_output_buffer(int nentries, bool fixedsize=false) {
        if (out) return -1;
        out = new FFBUFFER(nentries,fixedsize); 
        if (!out) return -1;
        myoutbuffer=true;
        return (out->init()?0:-1);
    }

    /** 
     *  \brief Assign the output channelname to a channel
     *
     * Attach the output of a \p ff_node to an existing channel, typically the input 
     * channel of another \p ff_node
     *
     *  \param o reference to a channel of type \p FFBUFFER
     *
     *  \return 0 if successful, -1 otherwise
     */
    virtual int set_output_buffer(FFBUFFER * const o) {
        if (myoutbuffer) return -1;
        out = o;
        return 0;
    }

    /** 
     *  \brief Assign the input channelname to a channel
     *
     * Attach the input of a \p ff_node to an existing channel, typically the output 
     * channel of another \p ff_node
     *
     *  \param i a buffer object of type \p FFBUFFER
     *
     *  \return 0 if successful, -1 otherwise
     */
    virtual int set_input_buffer(FFBUFFER * const i) {
        if (myinbuffer) return -1;
        in = i;
        return 0;
    }

    virtual inline int set_input(svector<ff_node *> & w) { return -1;}
    virtual inline int set_input(ff_node *) { return -1;}
    virtual inline bool isMultiInput() const { return multiInput;}
    virtual inline void setMultiInput()      { multiInput = true; }
    virtual inline int set_output(svector<ff_node *> & w) { return -1;}
    virtual inline int set_output(ff_node *) { return -1;}
    virtual inline bool isMultiOutput() const { return multiOutput;}
    virtual inline void setMultiOutput()      { multiOutput = true; }

    virtual inline void get_out_nodes(svector<ff_node*>&w) {}

    /**
     * \brief Run the ff_node
     *
     * \return 0 success, -1 otherwise
     */
    virtual int run(bool=false) { 
        thread = new thWorker(this);
        if (!thread) return -1;
        return thread->run();
    }
    
    /**
     * \internal
     * \brief Suspend (freeze) the ff_node and run it
     *
     * Only initialisation will be performed
     *
     * \return 0 success, -1 otherwise
     */
    virtual int freeze_and_run(bool=false) {
        thread = new thWorker(this);
        if (!thread) return 0;
        freeze();
        return thread->run();
    }

    /**
     * \brief Wait ff_node termination
     *
     * \return 0 success, -1 otherwise
     */
    virtual int  wait() { 
        if (!thread) return 0;
        return thread->wait(); 
    }
    
    /**
     * \brief Wait the freezing state
     *
     * It will happen on EOS arrival on the input channel
     *
     * \return 0 success, -1 otherwise
     */
    virtual int  wait_freezing() { 
        if (!thread) return 0;
        return thread->wait_freezing(); 
    }
    
    virtual void stop() {
        if (!thread) return; 
        thread->stop(); 
    }
    
    /**
     * \internal
     * \brief Freeze (suspend) a ff_node
     */
    virtual void freeze() { 
        if (!thread) return; 
        thread->freeze(); 
    }
    
    /**
     * \internal
     * \brief Thaw (resume) a ff_node
     */
    virtual void thaw(bool _freeze=false) { 
        if (!thread) return; 
        thread->thaw(_freeze);
    }
    
    /**
     * \internal
     * \brief Checks if a ff_node is frozen
     * \return \p true is it frozen
     */
    virtual bool isfrozen() const { 
        if (!thread) return false;
        return thread->isfrozen();
    }


    virtual bool isoutbuffermine() const { return myoutbuffer;}

    virtual int  cardinality(BARRIER_T * const b) { 
        barrier = b;
        return 1;
    }

    virtual void set_barrier(BARRIER_T * const b) {
        barrier = b;
    }

    /**
     * \brief Misure \ref ff::ff_node execution time
     *
     * \return time (ms)
     */
    virtual double ffTime() {
        return diffmsec(tstop,tstart);
    }

    /**
     * \brief Misure \ref ff_node::svc execution time
     *
     * \return time (ms)
     */
    virtual double wffTime() {
        return diffmsec(wtstop,wtstart);
    }

public:



    // TOGLIERE
    virtual bool iswaiting() const { 
        if (!thread) return false;
        return thread->iswaiting();
    }



    
    /*
     * \brief Default retry delay in nonblocking get/put on channels
     */
    enum {TICKS2WAIT=1000};

    /**
     * \brief The service callback (should be filled by user with parallel activity business code)
     *
     * \param task is a the input data stream item pointer (task)
     * \return output data stream item pointer
     */
    virtual void* svc(void * task) = 0;
    
    /**
     * \brief Service initialisation
     *
     * Called after run-time initialisation (e.g. thread spawning) but before 
     * to start to get items from input stream (can be useful for initialisation
     * of parallel activities, e.g. manual thread pinning that cannot be done in
     * the costructor because threads stil do not exist).
     *
     * \return 0
     */
    virtual int svc_init() { return 0; }
    
     /**
     *
     * \brief Service finalisation
     *
     * Called after EOS arrived (logical termination) but before shutdding down
     * runtime support (can be useful for housekeeping)
     */
    virtual void  svc_end() {}

    virtual void eosnotify(ssize_t id=-1) {}
    
    virtual int get_my_id() const { return myid; };
    
    /**
     * \internal
     * \brief Force ff_node-to-core pinning
     *
     * \param cpuID is the ID of the CPU to which the thread will be pinned.
     */
    virtual void  setAffinity(int cpuID) { 
        if (cpuID<0 || !threadMapper::instance()->checkCPUId(cpuID) ) {
            error("setAffinity, invalid cpuID\n");
        }
        CPUId=cpuID;
    }
    
    /** 
     * \internal
     * \brief Gets the CPU id (if set) of this node is pinned
     *
     * It gets the ID of the CPU where the ff_node is running.
     *
     * \return The identifier of the CPU.
     */
    virtual int getCPUId() const { return CPUId; }

#if defined(TEST_QUEUE_LOCK)
    virtual bool put(void * ptr) { 
        pthread_mutex_lock(&mutex_in);
        while(!in->push(ptr)){
            pthread_cond_wait(&queue_in_task_out,&mutex_in);
        }
        pthread_cond_signal(&queue_in_task_in);
        pthread_mutex_unlock(&mutex_in);
        return true;
    }

    virtual bool  get(void **ptr) { 
        pthread_mutex_lock(&mutex_out);
        while(!out->pop(ptr)) {
            pthread_cond_wait(&queue_out_task_in,&mutex_out);
        }
        pthread_cond_signal(&queue_out_task_out);
        pthread_mutex_unlock(&mutex_out);
        return true;
    }
#else // TEST_QUEUE_LOCK
    /**
     * \brief Nonblocking put onto output channel
     *
     * Wait-free and fence-free (under TSO)
     *
     * \param ptr is a pointer to the task
     *
     */
    virtual inline bool  put(void * ptr) { 
        return in->push(ptr);
    }
    
    /**
     * \brief Noblocking pop from input channel
     *
     * Wait-free and fence-free (under TSO)
     *
     * \param ptr is a pointer to the task
     *
     */
    virtual inline bool  get(void **ptr) { return out->pop(ptr);}
#endif
    
   
    virtual inline void losetime_out(void) {
        FFTRACE(lostpushticks+=ff_node::TICKS2WAIT; ++pushwait);
        ticks_wait(ff_node::TICKS2WAIT);
    }

    
    virtual inline void losetime_in(void) {
        FFTRACE(lostpopticks+=ff_node::TICKS2WAIT; ++popwait);
        ticks_wait(ff_node::TICKS2WAIT);
    }

    /**
     * \brief Gets input channel
     *
     * It returns a pointer to the input buffer.
     *
     * \return A pointer to the input buffer
     */
    virtual FFBUFFER * get_in_buffer() const { return in;}

    /**
     * \brief Gets pointer to the output channel
     *
     * It returns a pointer to the output buffer.
     *
     * \return A pointer to the output buffer.
     */
    virtual FFBUFFER * get_out_buffer() const { return out;}

    virtual const struct timeval getstarttime() const { return tstart;}

    virtual const struct timeval getstoptime()  const { return tstop;}

    virtual const struct timeval getwstartime() const { return wtstart;}

    virtual const struct timeval getwstoptime() const { return wtstop;}    

    virtual inline void svc_createOCL()  {} 

    virtual inline void svc_releaseOCL() {}

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

    /**
     * \brief Sends out the task
     *
     * It allows to emit tasks on output stream without returning from the \p svc method.
     * Make the ff_node to emit zero or more tasks per input task
     *
     * \param task a pointer to the task
     * \param retry number of tries to put (nonbloking partial) the task to output channel
     * \param ticks delay between successive retries
     * 
     */
    virtual bool ff_send_out(void * task, 
                             unsigned int retry=((unsigned int)-1),
                             unsigned int ticks=(TICKS2WAIT)) { 
        if (callback) return  callback(task,retry,ticks,callback_arg);

        for(unsigned int i=0;i<retry;++i) {
            if (push(task)) return true;
            FFTRACE(lostpushticks+=ticks; ++pushwait);
            ticks_wait(ticks);
        }     
        return false;
    }

    // Warning resetting queues while the node is running may produce unexpected results.
    virtual void reset() {
        if (in)  in->reset();
        if (out) out->reset();
    }

    virtual void registerEndCallback(void(*cb)(void*), void *param=NULL) {
        assert(cb != NULL);
        end_callback=cb;
        end_callback_param = param;
    }

    //fftree stuff
    fftree *fftree_ptr;

protected:

    ff_node():fftree_ptr(NULL),in(0),out(0),myid(-1),CPUId(-1),
              myoutbuffer(false),myinbuffer(false),
              skip1pop(false), in_active(true), 
              multiInput(false), multiOutput(false), 
              thread(NULL),callback(NULL),barrier(NULL),
              end_callback(NULL), end_callback_param(NULL) {
        time_setzero(tstart);time_setzero(tstop);
        time_setzero(wtstart);time_setzero(wtstop);
        wttime=0;
        FFTRACE(taskcnt=0;lostpushticks=0;pushwait=0;lostpopticks=0;popwait=0;ticksmin=(ticks)-1;ticksmax=0;tickstot=0);
#if defined(TEST_QUEUE_LOCK)
        
        pthread_mutex_init(&mutex_in,NULL);
        pthread_mutex_init(&mutex_out,NULL);
        pthread_cond_init(&queue_out_task_in, NULL);
        pthread_cond_init(&queue_out_task_out, NULL);
        pthread_cond_init(&queue_in_task_in,  NULL);
        pthread_cond_init(&queue_in_task_out,  NULL);
#endif
    };
    
    /** 
     *  \brief Destructor
     */
    virtual  ~ff_node() {
        if (end_callback) end_callback(end_callback_param);
        if (in && myinbuffer) delete in;
        if (out && myoutbuffer) delete out;
        if (thread) delete thread;
    };
    
    virtual inline void input_active(const bool onoff) {
        if (in_active != onoff)
            in_active= onoff;
    }

private:
    

    void registerCallback(bool (*cb)(void *,unsigned int,unsigned int,void *), void * arg) {
        callback=cb;
        callback_arg=arg;
    }

    void  setCPUId(int id) { CPUId = id;}

    inline int   getTid() const { return thread->getTid();} 
    

    class thWorker: public ff_thread {
    public:

        thWorker(ff_node * const filter):
            ff_thread(filter->barrier),filter(filter) {}
        

        inline bool push(void * task) {
            /* NOTE: filter->push and not buffer->push because of the filter can be a dnode
             */
            while (! filter->push(task)) filter->losetime_out();
            return true;
        }
        
        inline bool pop(void ** task) {
            /* 
             * NOTE: filter->pop and not buffer->pop because of the filter can be a dnode
             */
            while (! filter->pop(task)) {
                if (!filter->in_active) { *task=NULL; return false;}
                filter->losetime_in();
            } 
            return true;
        }

        inline bool put(void * ptr) { return filter->put(ptr);}

        inline bool get(void **ptr) { return filter->get(ptr);}

        inline void svc_createOCL()  { filter-> svc_createOCL();}
        
        inline void svc_releaseOCL() { filter-> svc_releaseOCL();}

        void* svc(void * ) {
            void * task = NULL;
            void * ret = (void*)FF_EOS;
            bool inpresent  = (filter->get_in_buffer() != NULL);
            bool outpresent = (filter->get_out_buffer() != NULL);
            bool skipfirstpop = filter->skipfirstpop(); 
            bool exit=false;            

            gettimeofday(&filter->wtstart,NULL);
            do {
                svc_createOCL();

                if (inpresent) {
                    if (!skipfirstpop) pop(&task); 
                    else skipfirstpop=false;
                    if ((task == EOS) || 
                        (task == EOS_NOFREEZE)) {
                        ret = task;
                        filter->eosnotify();
                        if (outpresent && (task == (void*)FF_EOS))  push(task); // only EOS is propagated
                        break;
                    }
                    if (task == GO_OUT) break;
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
                if (ret == GO_OUT) break;     
                if (!ret || (ret == EOS) || (ret == EOS_NOFREEZE)) {
                    // NOTE: The EOS is gonna be produced in the output queue
                    // and the thread exits even if there might be some tasks
                    // in the input queue !!!
                    if (!ret) ret = EOS;
                    exit=true;
                }
                if (outpresent && (ret != GO_ON && ret != EOS_NOFREEZE)) push(ret);
            } while(!exit);
            
            gettimeofday(&filter->wtstop,NULL);
            filter->wttime+=diffmsec(filter->wtstop,filter->wtstart);
            
            return ret;
        }
        
        int svc_init() {
#if !defined(HAVE_PTHREAD_SETAFFINITY_NP) && !defined(NO_DEFAULT_MAPPING)
            int cpuId = filter->getCPUId();  
            if (ff_mapThreadToCpu((cpuId<0) ? (cpuId=threadMapper::instance()->getCoreId(tid)) : cpuId)!=0)
                error("Cannot map thread %d to CPU %d, mask is %u,  size is %u,  going on...\n",tid, (cpuId<0) ? threadMapper::instance()->getCoreId(tid) : cpuId, threadMapper::instance()->getMask(), threadMapper::instance()->getCListSize());            
            filter->setCPUId(cpuId);
#endif
            gettimeofday(&filter->tstart,NULL);
            return filter->svc_init(); 
        }
        
        virtual void svc_end() {
            filter->svc_end();
            gettimeofday(&filter->tstop,NULL);
        }
        
        int run() { 
            int CPUId = ff_thread::spawn(filter->getCPUId());             
            filter->setCPUId(CPUId);
            return (CPUId==-2)?-1:0;
        }

        virtual inline int wait() { return ff_thread::wait();}

        virtual inline int wait_freezing() { return ff_thread::wait_freezing();}

        virtual inline void freeze() { ff_thread::freeze();}

        bool isfrozen() const { return ff_thread::isfrozen();}

        int get_my_id() const { return filter->get_my_id(); };
        
    protected:    
        ff_node * const filter;
    };

private:
    FFBUFFER        * in;           ///< Input buffer, built upon SWSR lock-free (wait-free) 
                                    ///< (un)bounded FIFO queue                                 
    FFBUFFER        * out;          ///< Output buffer, built upon SWSR lock-free (wait-free) 
                                    ///< (un)bounded FIFO queue 
    int               myid;         ///< This is the node id, it is valid only for farm's workers
    int               CPUId;    
    bool              myoutbuffer;
    bool              myinbuffer;
    bool              skip1pop;
    bool              in_active;    // allows to disable/enable input tasks receiving   
    bool              multiInput;   // if the node is a multi input node this is true
    bool              multiOutput;  // if the node is a multi output node this is true
    thWorker        * thread;       /// A \p thWorker object, which extends the \p ff_thread class 
    bool (*callback)(void *,unsigned int,unsigned int, void *);
    void            * callback_arg;
    BARRIER_T       * barrier;      /// A \p Barrier object
    void (*end_callback)(void*);  
    void *end_callback_param;
    struct timeval tstart;
    struct timeval tstop;
    struct timeval wtstart;
    struct timeval wtstop;
    double wttime;

#if defined(TEST_QUEUE_LOCK)
    pthread_mutex_t mutex_in, mutex_out;
    pthread_cond_t  queue_out_task_out, queue_out_task_in;
    pthread_cond_t  queue_in_task_out, queue_in_task_in;
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

/* just a node interface for the input and output buffers */
struct ff_buffernode: ff_node {
    ff_buffernode() {}
    ff_buffernode(int nentries, bool fixedsize=true, int id=-1) {
        set_id(id);
        if (create_input_buffer(nentries,fixedsize)<0) abort();
        set_output_buffer(ff_node::get_in_buffer());
    }
    ff_buffernode(int id, FFBUFFER *in, FFBUFFER *out) {
        set_id(id);
        set_input_buffer(in);
        set_output_buffer(out);
    }
    void* svc(void*){return NULL;}
    void set(int nentries, bool fixedsize=true, int id=-1) {
        set_id(id);
        if (create_input_buffer(nentries,fixedsize) < 0) abort();
        set_output_buffer(ff_node::get_in_buffer());
    }

    template<typename T>
    inline bool  put(T *ptr) {  return ff_node::put(ptr);  }
};


/*!
 *  \class ff_node_t
 *  \ingroup building_blocks
 *
 *  \brief The FastFlow typed abstract contanier for a parallel activity (actor).
 *
 *  Key method is: \p svc (pure virtual).
 *
 *  This class is defined in \ref node.hpp
 */

template<typename T>
struct ff_node_t:ff_node {
    ff_node_t():
        GO_ON((T*)FF_GO_ON),
        EOS((T*)FF_EOS),
        GO_OUT((T*)FF_GO_OUT),
        EOS_NOFREEZE((T*)FF_EOS_NOFREEZE) {}
    T *GO_ON, *EOS, *GO_OUT, *EOS_NOFREEZE;
    virtual ~ff_node_t()  {}
    void *svc(void *task) { return svc(reinterpret_cast<T*>(task));};
    virtual T* svc(T*)=0;
};

} // namespace ff

#endif /* FF_NODE_HPP */
