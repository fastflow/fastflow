/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \file farm.hpp
 *  \ingroup high_level_patterns core_patterns building_blocks
 *  \brief Farm pattern
 *
 *  It works on a stream of tasks. Workers are non-blocking threads
 *  not tasks. It is composed by: Emitter (E), Workers (W), Collector (C).
 *  They all are C++ objects.
 *  Overall, it has one (optional) input stream and one (optional) output stream.
 *  Emitter gets stream items (tasks, i.e. C++ objects) and disptach them to 
 *  Workers (activating svc method). On svn return (or ff_send_out call), tasks
 *  are sent to Collector that gather them and output them in the output stream.

 *  Dispatching policy can be configured in the Emitter. Gathering policy in the
 *  Collector.
 * 
 *  In case of no output stream the Collector is usually not needed. Emitter 
 *  should always exist, even with no input stream.
 * 
 *  There exists several variants of the farm pattern, including
 * 
 *  \li Master-worker: no collector, tasks from Workers return to Emitter
 *  \li Ordering farm: default emitter and collector, tasks are gathered
 *  in the same order they are dispatched
 * 
 * \todo Includes classes at different levels. To be split sooner or later.
 * High level farm function to be wrapped in a separate class.
 */

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
 
#ifndef FF_FARM_HPP
#define FF_FARM_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <ff/platforms/platform.h>
#include <ff/lb.hpp>
#include <ff/gt.hpp>
#include <ff/node.hpp>

#if defined( HAS_CXX11_VARIADIC_TEMPLATES )
#include <functional>
#endif

namespace ff {

/* ---------------- basic high-level macros ----------------------- */

    /* FF_FARMA creates an accelerator farm with 'nworkers' workers. 
     * The functions called in each worker must have the following 
     * signature:
     *   bool (*F)(type &)
     * If the function F returns false an EOS is produced in the 
     * output channel.
     */

#define FF_FARMA(farmname, F, type, nworkers, qlen)                     \
    class _FF_node_##farmname_##F: public ff_node {                     \
    public:                                                             \
     void* svc(void* task) {                                            \
        type* _ff_in = (type*)task;                                     \
        if (!F(*_ff_in)) return NULL;                                   \
        return task;                                                    \
     }                                                                  \
    };                                                                  \
    ff_farm<> _FF_node_##farmname(true,qlen,qlen,true,nworkers,true);   \
    _FF_node_##farmname.cleanup_workers();                              \
    _FF_node_##farmname.set_scheduling_ondemand();                      \
    std::vector<ff_node*> w_##farmname;                                 \
    for(int i=0;i<nworkers;++i)                                         \
        w_##farmname.push_back(new _FF_node_##farmname_##F);            \
    _FF_node_##farmname.add_workers(w_##farmname);                      \
    _FF_node_##farmname.add_collector(NULL);

#define FF_FARMARUN(farmname)  _FF_node_##farmname.run()
#define FF_FARMAOFFLOAD(farmname, task)  _FF_node_##farmname.offload(task);
#define FF_FARMAGETRESULTNB(farmname, result) _FF_node_##farmname.load_result_nb((void**)result)
#define FF_FARMAGETRESULT(farmname, result) _FF_node_##farmname.load_result((void**)result)
#define FF_FARMAWAIT(farmname) _FF_node_##farmname.wait()
#define FF_FARMAEND(farmname)  _FF_node_##farmname.offload(EOS)

/* ---------------------------------------------------------------- */


/*!
 *  \class ff_farm
 * \ingroup  high_level_patterns core_patterns
 *
 *  \brief The Farm skeleton, with Emitter (\p lb_t) and Collector (\p gt_t).
 *
 *  The Farm skeleton can be seen as a 3-stages pipeline. The first stage is
 *  the \a Emitter (\ref ff_loadbalancer "lb_t") that act as a load-balancer;
 *  the last (optional) stage would be the \a Collector (\ref ff_gatherer
 *  "gt_t") that gathers the results computed by the \a Workers, which are
 *  ff_nodes.
 *
 *  This class is defined in \ref farm.hpp
 */
template<typename lb_t=ff_loadbalancer, typename gt_t=ff_gatherer>
class ff_farm: public ff_node {
protected:
    inline int cardinality(BARRIER_T * const barrier)  { 
        int card=0;
        for(size_t i=0;i<workers.size();++i) 
            card += workers[i]->cardinality(barrier);
        
        lb->set_barrier(barrier);
        if (gt) gt->set_barrier(barrier);

        return (card + 1 + ((collector && !collector_removed)?1:0));
    }

    inline int prepare() {
        size_t nworkers = workers.size();
        for(size_t i=0;i<nworkers;++i) {
            if (workers[i]->create_input_buffer((int) (ondemand ? ondemand: (in_buffer_entries/nworkers + 1)), 
                                                (ondemand ? true: fixedsize))<0) return -1;
            if (collector || lb->masterworker() || collector_removed) 
                // NOTE: force unbounded queue if masterworker
                if (workers[i]->create_output_buffer((int) (out_buffer_entries/nworkers + DEF_IN_OUT_DIFF), 
                                                     (lb->masterworker()?false:fixedsize))<0)
                    return -1;
            lb->register_worker(workers[i]);
            if (collector && !collector_removed) gt->register_worker(workers[i]);
        }
        prepared=true;
        return 0;
    }

    int freeze_and_run(bool=false) {
        if (!prepared) if (prepare()<0) return -1;
        freeze();
        return run(true);
    } 

    // template<typename T>
    // struct ff_node_F: public ff_node {
    //     typedef T*(*F_t)(T*,ff_node*const);
    //     ff_node_F(F_t F):F(F) {};
    //     inline void *svc(void *t) { return F((T*)t,this); }
    //     F_t F;
    // };

#if defined( HAS_CXX11_VARIADIC_TEMPLATES )
    // NOTE: std::function can introduce a bit of extra overhead. 
    template<typename T>
    struct ff_node_F: public ff_node {
        ff_node_F(std::function<T*(T*,ff_node*const)> F):F(F) {};
        inline void *svc(void *t) { return F((T*)t,this); }
        std::function<T*(T*,ff_node*const)> F;
    };
#endif

    inline void skipfirstpop(bool sk)   { 
        if (sk) lb->skipfirstpop();
        skip1pop=sk;
    }

    
public:
    /*
     *TODO
     */
    enum { DEF_MAX_NUM_WORKERS=(MAX_NUM_THREADS-2), DEF_IN_BUFF_ENTRIES=2048, DEF_IN_OUT_DIFF=128, 
           DEF_OUT_BUFF_ENTRIES=(DEF_IN_BUFF_ENTRIES+DEF_IN_OUT_DIFF)};

    typedef lb_t LoadBalancer_t;

    typedef gt_t Gatherer_t;

    /**                                                                                                                                 
     * \ingroup high_level_patterns
     * \brief High-level pattern constructor
     */
#if defined( HAS_CXX11_VARIADIC_TEMPLATES )
    template<typename T>
    ff_farm(const std::function<T*(T*,ff_node*const)> &F, int nw, bool input_ch=false):
        has_input_channel(input_ch),prepared(false),collector_removed(false),ondemand(0),
        in_buffer_entries(DEF_IN_BUFF_ENTRIES),
        out_buffer_entries(DEF_OUT_BUFF_ENTRIES),
        worker_cleanup(true),
        max_nworkers(nw),
        emitter(NULL),collector(NULL),
        lb(new lb_t(nw)),gt(new gt_t(nw)),
        workers(nw),fixedsize(false) {

        std::vector<ff_node*> w(nw);        
        for(int i=0;i<nw;++i) w[i]=new ff_node_F<T>(F);
        add_workers(w);
        add_collector(NULL);
        if (has_input_channel) { 
            if (create_input_buffer(in_buffer_entries, fixedsize)<0) {
                error("FARM, creating input buffer\n");
            }
        }
    }
#endif
    /**
     * \ingroup core_patterns
     * @brief Core patterns constructor 2
     *
     * This is a constructor at core patterns level
     * @param W vector of workers
     * @param Emitter pointer to Emitter object (mandatory)
     * @param Collector pointer to Collector object (optional)
     * @param input_ch \p true for enabling the input stream
     */
    ff_farm(std::vector<ff_node*>& W, ff_node *const Emitter=NULL, ff_node *const Collector=NULL, bool input_ch=false):
        has_input_channel(input_ch),prepared(false),collector_removed(false),ondemand(0),
        in_buffer_entries(DEF_IN_BUFF_ENTRIES),
        out_buffer_entries(DEF_OUT_BUFF_ENTRIES),
        worker_cleanup(false),
        max_nworkers(W.size()),
        emitter(NULL),collector(NULL),
        lb(new lb_t(W.size())),gt(new gt_t(W.size())),
        workers(W.size()),fixedsize(false) {

        assert(W.size()>0);
        add_workers(W);

        if (Emitter) add_emitter(Emitter); 

        // add default collector even if Collector is NULL, 
        // if you don't want the collector you have to call remove_collector
        add_collector(Collector); 

        if (has_input_channel) { 
            if (create_input_buffer(in_buffer_entries, fixedsize)<0) {
                error("FARM, creating input buffer\n");
            }
        }
    }

    /**
     * \ingroup core_patterns
     * \brief Core patterns constructor 1
     *
     *  This is a constructor at core patterns level. To be coupled with \p add_worker, \p add_emitter, and \p add_collector
     *
     *  \param input_ch = true to set accelerator mode
     *  \param in_buffer_entries = input queue length
     *  \param out_buffer_entries = output queue length
     *  \param max_num_workers = highest number of farm's worker
     *  \param worker_cleanup = true deallocate worker object at exit
     *  \param fixedsize = true uses only fixed size queue
     */
    ff_farm(bool input_ch=false,
            int in_buffer_entries=DEF_IN_BUFF_ENTRIES, 
            int out_buffer_entries=DEF_OUT_BUFF_ENTRIES,
            bool worker_cleanup=false, // NOTE: by default no cleanup at exit is done !
            int max_num_workers=DEF_MAX_NUM_WORKERS,
            bool fixedsize=false):  // NOTE: by default all the internal farm queues are unbounded !
        has_input_channel(input_ch),prepared(false),collector_removed(false),ondemand(0),
        in_buffer_entries(in_buffer_entries),
        out_buffer_entries(out_buffer_entries),
        worker_cleanup(worker_cleanup),
        max_nworkers(max_num_workers),
        emitter(NULL),collector(NULL),
        lb(new lb_t(max_num_workers)),gt(new gt_t(max_num_workers)),
        workers(max_num_workers),fixedsize(fixedsize) {
        for(int i=0;i<max_num_workers;++i) workers[i]=NULL;

        if (has_input_channel) { 
            if (create_input_buffer(in_buffer_entries, fixedsize)<0) {
                error("FARM, creating input buffer\n");
            }
        }
    }

    /** 
     * \brief Destructor
     *
     * Destruct the load balancer, the
     * gatherer, all the workers
     */
    virtual ~ff_farm() { 
        if (end_callback) { 
            end_callback(end_callback_param);
            end_callback = NULL;
        }
        if (lb) { delete lb; lb=NULL;}
        if (gt) { delete gt; gt=NULL;}
        if (worker_cleanup) {
            for(int i=0;i<max_nworkers; ++i) 
                if (workers[i]) delete workers[i];
        }
        for(size_t i=0;i<internalSupportNodes.size();++i) {
            delete internalSupportNodes.back();
            internalSupportNodes.pop_back();
        }

        if (barrier) {delete barrier; barrier=NULL;}
    }

    /** 
     *
     *  \brief Adds the emitter
     *
     *  It adds an Emitter to the Farm. The Emitter is of type \p ff_node and
     *  there can be only one Emitter in a Farm skeleton. 
     *  
     *  \param e the \p ff_node acting as an Emitter 
     *
     *  \return Returns 0 if successful -1 otherwise
     *
     */
    int add_emitter(ff_node * e) { 
        assert(e!=NULL);
        if (emitter) {
            error("FARM, add_emitter: emitter already present\n");
            return -1; 
        }
        if (e->isMultiInput()) {
            error("FARM, add_emitter: the node is a multi-input node, please do use ff_node together with ff_farm::setMultiInput() method\n");
            return -1;
        }
        emitter = e;

        if (in) {
            assert(has_input_channel);
            emitter->set_input_buffer(in);
        }
        if (lb->set_filter(emitter)) return -1;
        return 0;
    }

    /**
     *
     * \brief Set scheduling with on demand polity
     *
     * The default scheduling policy is round-robin, When there is a great
     * computational difference among tasks the round-robin scheduling policy
     * could lead to load imbalance in worker's workload (expecially with short
     * stream length). The on-demand scheduling policy can guarantee a near
     * optimal load balancing in lots of cases. Alternatively it is always
     * possible to define a complete application-level scheduling by redefining
     * the ff_loadbalancer class.
     *
     * \param inbufferentries sets the number of queue slot for one worker
     * threads.
     *
     */
    void set_scheduling_ondemand(const int inbufferentries=1) { 
        if (inbufferentries<=0) ondemand=1;
        else ondemand=inbufferentries;
    }

    /**
     *  \brief Adds workers to the form
     *
     *  Add workers to the Farm. There is a limit to the number of workers that
     *  can be added to a Farm. This limit is set by default to 64. This limit
     *  can be augmented by passing the desired limit as the fifth parameter of
     *  the \p ff_farm constructor.
     *
     *  \param w a vector of \p ff_nodes which are Workers to be attached to
     *  the Farm.
     *
     *  \return 0 if successsful, otherwise -1 is returned.
     */
    int add_workers(std::vector<ff_node *> & w) { 
        if ((workers.size()+w.size())> (size_t)max_nworkers) {
            error("FARM, try to add too many workers, please increase max_nworkers\n");
            return -1; 
        }
        if (w.size()==0) {
            error("FARM, try to add zero workers!\n");
            return -1; 
        }        
        for(size_t i=0;i<w.size();++i) {
            workers.push_back(w[i]);
			(workers.back())->set_id(int(i));
        }
        return 0;
    }

    /**
     *  \brief Adds the collector
     *
     *  It adds the Collector filter to the farm skeleton. If no object is
     *  passed as a colelctor, than a default collector will be added (i.e.
     *  \link ff_gatherer \endlink). Note that it is not possible to add more
     *  than one collector. 
     *
     *  \param c Collector object
     *  \param outpresent outstream?
     *
     *  \return The status of \p set_filter(x) if successful, otherwise -1 is
     *  returned.
     */
    int add_collector(ff_node * c, bool outpresent=false) { 
        if (collector) {
            error("add_collector: collector already defined!\n");
            return -1; 
        }
        if (!gt) return -1; 

        collector = ((c!=NULL)?c:(ff_node*)gt);
        
        if (has_input_channel) { /* it's an accelerator */
            if (create_output_buffer(out_buffer_entries, false)<0) return -1;
        }
        
        return gt->set_filter(c);
    }
    
    /**
     *
     * \brief Sets the feedback channel from the collector to the emitter
     *
     * This method allows to estabilish a feedback channel from the Collector
     * to the Emitter. If the collector is present, than the collector output
     * queue will be connected to the emitter input queue (feedback channel)
     *
     * \return 0 if successful, otherwise -1 is returned.
     *
     */
    int wrap_around(bool multi_input=false) {
        if (!collector || collector_removed) {
            if (lb->set_masterworker()<0) return -1;
            if (!multi_input && !has_input_channel) lb->skipfirstpop();
            return 0;
        }

        if (!multi_input) {
            if (create_input_buffer(in_buffer_entries, false)<0) {
                error("FARM, creating input buffer\n");
                return -1;
            }
            if (set_output_buffer(get_in_buffer())<0) {
                error("FARM, setting output buffer\n");
                return -1;
            }
            lb->skipfirstpop();
        } else {
            if (create_output_buffer(out_buffer_entries, false)<0) {
                error("FARM, creating output buffer for multi-input configuration\n");
                return -1;
            }
            internalSupportNodes.push_back(new ff_buffernode(0, NULL,get_out_buffer()));
            if (set_output_buffer(get_out_buffer())<0) {
                error("FARM, setting output buffer for multi-input configuration\n");
                return -1;
            }
            if (lb->set_internal_multi_input(internalSupportNodes)<0)
                return -1;
        }

        return 0;
    }

    /**
     *
     * \brief Removes the collector
     *
     * It allows not to start the collector thread, whereas all worker's output
     * buffer will be created as if it were present.
     *
     * \return 0 is always returned.
     */
    int remove_collector() { 
        collector_removed = true;
        return 0;
    }

    /**
     * \internal
     * \brief Sets multiple input nodes
     *
     * It sets multiple inputs to the node.
     *
     *
     * \return The status of \p set_input(x) otherwise -1 is returned.
     */
    inline int set_input(svector<ff_node *> & w) { 
        assert(isMultiInput());
        return lb->set_input(w);
    }

    inline int set_input(ff_node *node) { 
        assert(isMultiInput());
        return lb->set_input(node);
    }

    inline void setMultiInput() { 
        ff_node::setMultiInput();        
    }


    /**
     * \internal
     * \brief Delete workers when the destructor is called.
     *
     */
    void cleanup_workers() {
        worker_cleanup = true;
    }

    /**
     * \brief Execute the Farm 
     *
     * It executes the form.
     *
     * \param skip_init A booleon value showing if the initialization should be
     * skipped
     *
     * \return If successful 0, otherwise a negative is returned.
     *
     */
    int run(bool skip_init=false) {
        if (!skip_init) {
            // set the initial value for the barrier 

            if (!barrier)  barrier = new BARRIER_T;
            const int nthreads = cardinality(barrier);
            if (nthreads > MAX_NUM_THREADS) {
                error("FARM, too much threads, increase MAX_NUM_THREADS !\n");
                return -1;
            }

            barrier->barrierSetup(nthreads);            
        }
        
        if (!prepared) if (prepare()<0) return -1;

        if (lb->run()<0) {
            error("FARM, running load-balancer module\n");
            return -1;        
        }
        if (!collector_removed)
            if (collector && gt->run()<0) {
                error("FARM, running gather module\n");
                return -1;
            }

        return 0;
    }

    /** 
     * \brief Executs the farm and wait for workers to complete
     *
     * It executes the farm and waits for all workers to complete their
     * tasks.
     *
     * \return If successful 0, otherwise a negative value is returned.
     */
    virtual int run_and_wait_end() {
        if (isfrozen()) {
            stop();
            thaw();
            if (wait()<0) return -1;
            return 0;
        }
        stop();
        if (run()<0) return -1;
        if (wait()<0) return -1;
        return 0;
    }

    /** 
     * \brief Executes the farm and then freeze.
     *
     * It executs the form and then freezes the form.
     * If workers are frozen, it is possible to wake up just a subset of them.
     *
     * \return If successful 0, otherwise a negative value
     */
    virtual int run_then_freeze(ssize_t nw=-1) {
        if (isfrozen()) {
            // true means that next time threads are frozen again
            thaw(true, nw); 
            return 0;
        }
        if (!prepared) if (prepare()<0) return -1;
        freeze();
        return run();
    }
    
    /** 
     * \brief Puts the thread in waiting state
     *
     * It puts the thread in waiting state.
     *
     * \return 0 if successful, otherwise -1 is returned.
     */
    int wait(/* timeval */ ) {
        int ret=0;
        if (lb->wait()<0) ret=-1;
        if (!collector_removed && collector) if (gt->wait()<0) ret=-1;
        return ret;
    }

    /** 
     * \brief Waits for freezing
     *
     * It waits for thread to freeze.
     *
     * \return 0 if successful otherwise -1 is returned.
     */
    inline int wait_freezing(/* timeval */ ) {
        int ret=0;
        if (lb->wait_freezing()<0) ret=-1;
        if (!collector_removed && collector) if (gt->wait_freezing()<0) ret=-1;
        return ret; 
    } 

    /** 
     * \internal
     * \brief Forces a thread to Stop
     *
     * It forces the thread to stop at the next EOS signal. 
     */
    inline void stop() {
        lb->stop();
        if (collector) gt->stop();
    }

    /** 
     * \internal
     * \brief Forces the thread to freeze at next FF_EOS.
     *
     * It forces a thread to Freeze itself.
     */
    inline void freeze() {
        lb->freeze();
        if (collector) gt->freeze();
    }

    /**
     * \internal
     * \brief Thaws the thread
     *
     * If the thread is frozen, then thaw it. 
     */
    inline void thaw(bool _freeze=false, ssize_t nw=-1) {
        lb->thaw(_freeze, nw);
        if (collector) gt->thaw(_freeze, nw);
    }

    /**
     * \internal
     * \brief Checks if the Farm is frozen
     *
     * It checks if the farm is frozen.
     *
     * \return The status of \p isfrozen().
     */
    inline bool isfrozen() const { return lb->isfrozen(); }

    /**
     * \breif Offloads teh task to farm
     *
     * It offloads the given task to the farm.
     *
     * \param task is a void pointer
     * \param retry showing the number of tries to offload
     * \param ticks is the number of ticks to wait
     *
     * \return \p true if successful, otherwise \p false
     */
    inline bool offload(void * task,
                        unsigned int retry=((unsigned int)-1),
                        unsigned int ticks=ff_loadbalancer::TICKS2WAIT) { 
        FFBUFFER * inbuffer = get_in_buffer();

        if (inbuffer) {
            for(unsigned int i=0;i<retry;++i) {
                if (inbuffer->push(task)) return true;
                ticks_wait(ticks);
            }     
            return false;
        }
        
        if (!has_input_channel) 
            error("FARM: accelerator is not set, offload not available");
        else
            error("FARM: input buffer creation failed");
        return false;
    }

    /**
     * \brief Loads results into gatherer
     *
     * It loads the results from the gatherer (if any).
     *
     * \param task is a void pointer
     * \param retry is the number of tries to load the results
     * \param ticks is the number of ticks to wait
     *
     * \return \p false if EOS arrived or too many retries, \p true if  there is a new value
     */
    inline bool load_result(void ** task,
                            unsigned int retry=((unsigned int)-1),
                            unsigned int ticks=ff_gatherer::TICKS2WAIT) {
        if (!collector) return false;
        for(unsigned int i=0;i<retry;++i) {
            if (gt->pop_nb(task)) {
                if ((*task != (void *)FF_EOS)) return true;
                else return false;
            }
            ticks_wait(ticks);
        }
        return false;
    }

    /**
     * \brief Loads result with non-blocking
     *
     * It loads the result with non-blocking situation.
     *
     * \param task is a void pointer
     *
     * \return \false if no task is present, otherwise \true if there is a new
     * value. It should be checked if the task has a \p FF_EOS
     *
     */
    inline bool load_result_nb(void ** task) {
        if (!collector) return false;
        return gt->pop_nb(task);
    }
    
    /**
     * \internal
     * \brief Gets lb (Emitter) node
     *
     * It gets the lb node (Emitter)
     *
     * \return A pointer to the load balancer is returned.
     *
     */
    inline lb_t * getlb() const { return lb;}

    /**
     * \internal
     * \brief Gets gt (Collector) node
     *
     * It gets the gt node (collector)
     *
     * \return A pointer to the gatherer is returned.
     */
    inline gt_t * getgt() const { return gt;}

    /**
     * \internal
     * \brief Gets workers list
     *
     * It gets the list of the workers
     *
     * \return A list of workers is returned.
     */
    const svector<ff_node*>& getWorkers() const { return workers; }


    /**
     * \internal
     * \brief Gets the number of workers
     *
     * The number of workers is returned.
     *
     * \return An integet value showing the number of workers.
     */
    size_t getNWorkers() const { return workers.size();}

    inline void get_out_nodes(svector<ff_node*>&w) {
        if (collector && !collector_removed) {
            if ((ff_node*)gt == collector) {
                ff_node *outnode = new ff_buffernode(-1, NULL,gt->get_out_buffer());
                internalSupportNodes.push_back(outnode);
            } else {
                collector->get_out_nodes(w);
                if (w.size()==0) w.push_back(collector);
            }
            return;
        }
        for(size_t i=0;i<workers.size();++i)
            workers[i]->get_out_nodes(w);
        if (w.size()==0) w = workers;
    }

    /**
     * \internal
     * \brief Resets input/output queues.
     * 
     *  Warning resetting queues while the node is running may produce unexpected results.
     */
    void reset() {
        if (lb)  lb->reset();
        if (gt)  gt->reset();
        for(size_t i=0;i<workers.size();++i) workers[i]->reset();
    }

    /**
     * \internal
     * \brief Gets the starting time
     *
     * It returns the starting time.
     *
     * \return A structure of \p timeval showing the starting time.
     *
     */
    const struct timeval getstarttime() const { return lb->getstarttime();}

    /**
     * \internal
     * \brief Gets the stoping time
     *
     * It returns the structure showing the finishing time. It
     * is the collector then return the finishing time of the farm. otherwise,
     * collects the finishing time in all workers and add them in a vector and
     * then return the vector, showing the collective finishing time of the
     * farm with no collector.
     *
     * \return A \timeval showing the finishing time of the farm.
     */
    const struct timeval  getstoptime()  const {
        if (collector) return gt->getstoptime();
        const struct timeval zero={0,0};
        std::vector<struct timeval > workertime(workers.size()+1,zero);
        for(size_t i=0;i<workers.size();++i)
            workertime[i]=workers[i]->getstoptime();
        workertime[workers.size()]=lb->getstoptime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }

    /**
     * \internal
     * \brief Gets the starting time
     *
     * It returnes the starting time.
     *
     * \return A struct of type timeval showing the starting time.
     */
    const struct timeval  getwstartime() const { return lb->getwstartime(); }    

    /**
     * \internal
     * \brief Gets the finishing time
     *
     * It returns the finishing time if there exists a collector in the farm.
     * If there is no collector, then the finishing time of individual workers
     * is collected in the form of a vector and return that vector.
     *
     * \return The vector showing the finishing time.
     */
    const struct timeval  getwstoptime() const {
        if (collector) return gt->getwstoptime();
        const struct timeval zero={0,0};
        std::vector<struct timeval > workertime(workers.size()+1,zero);
        for(size_t i=0;i<workers.size();++i) {
            workertime[i]=workers[i]->getwstoptime();
        }
        workertime[workers.size()]=lb->getwstoptime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }
    
    /**
     * \internal
     * \brief Gets the time spent in \p svc_init
     *
     * The returned time comprises the time spent in \p svc_init and in \p
     * svc_end methods.
     *
     * \return A double value showing the time taken in \p svc_init
     */
    double ffTime() {
        if (collector)
            return diffmsec(gt->getstoptime(), lb->getstarttime());
        return diffmsec(getstoptime(),lb->getstarttime());
    }

    /**
     * \internal
     * \brief Gets the time spent in \p svc
     *
     * The returned time considers only the time spent in the svc methods.
     *
     * \return A double value showing the time taken in \p svc.
     */
    double ffwTime() {
        if (collector)
            return diffmsec(gt->getwstoptime(), lb->getwstartime());

        return diffmsec(getwstoptime(),lb->getwstartime());
    }


#if defined(TRACE_FASTFLOW)
    void ffStats(std::ostream & out) { 
        out << "--- farm:\n";
        lb->ffStats(out);
        for(size_t i=0;i<workers.size();++i) workers[i]->ffStats(out);
        if (collector) gt->ffStats(out);
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }
#endif

protected:

    /**
     * \brief svc method
     */
    void* svc(void * task) { return NULL; }

    /**
     * \brief The svc_init method
     */
    int svc_init()       { return -1; };

    /**
     * \brief The svc_end method
     */
    void svc_end()        {}

    int get_my_id() const { return -1; };


    void setAffinity(int) { 
        error("FARM, setAffinity: cannot set affinity for the farm\n");
    }

    int getCPUId() const { return -1;}

    /** 
     *  \brief Creates the input buffer for the emitter node
     *
     *  This function redefines the ff_node's virtual method of the same name.
     *  It creates an input buffer for the Emitter node. 
     *
     *  \param nentries the size of the buffer
     *  \param fixedsize flag to decide whether the buffer is resizable. 
     *
     *  \return If successful 0, otherwsie a negative value.
     */
    int create_input_buffer(int nentries, bool fixedsize) {
        if (in) {
            error("FARM create_input_buffer, buffer already present\n");
            return -1;
        }
        if (emitter) {
            if (emitter->create_input_buffer(nentries,fixedsize)<0) return -1;
            in = emitter->get_in_buffer();
        } else {
            if (ff_node::create_input_buffer(nentries, fixedsize)<0) return -1;
        }
        lb->set_in_buffer(in);

        // old code
        //if (ff_node::create_input_buffer(nentries, fixedsize)<0) return -1;
        //lb->set_in_buffer(in);

        return 0;
    }
    
    /**
     * \internal
     * \brief Creates the output buffer for the collector
     *
     *  This function redefines the ff_node's virtual method of the same name.
     *  It create an output buffer for the Collector
     *
     *  \param nentries the size of the buffer
     *  \param fixedsize flag to decide whether the buffer is resizable. 
     *  Default is \p false
     *
     *  \return If successful 0, otherwise a negative value.
     */
    int create_output_buffer(int nentries, bool fixedsize=false) {
        if (!collector && !collector_removed) {
            error("FARM with no collector, cannot create output buffer\n");
            return -1;
        }        
        if (out) {
            error("FARM create_output_buffer, buffer already present\n");
            return -1;
        }
        if (ff_node::create_output_buffer(nentries, fixedsize)<0) return -1;
        gt->set_out_buffer(out);
        if (collector && ((ff_node*)gt != collector)) collector->set_output_buffer(out);
        return 0;
    }

    /**
     *
     *  \brief Sets the output buffer of the collector 
     *
     *  This function redefines the ff_node's virtual method of the same name.
     *  Set the output buffer for the Collector.
     *
     *  \param o a buffer object, which can be of type \p SWSR_Ptr_Buffer or 
     *  \p uSWSR_Ptr_Buffer
     *
     *  \return 0 if successful, otherwise -1 is returned.
     */
    int set_output_buffer(FFBUFFER * const o) {
        if (!collector && !collector_removed) {
            error("FARM with no collector, cannot set output buffer\n");
            return -1;
        }
        gt->set_out_buffer(o);
        if (collector && ((ff_node*)gt != collector)) collector->set_output_buffer(o);
        return 0;
    }

    /**
     * \brief Gets Emitter
     *
     * It returns a pointer to the emitter.
     *
     * \return A pointer of the FastFlow node which is actually the emitter.
     */
    ff_node* getEmitter()   { return emitter;}

    /**
     * \brief Gets Collector
     * 
     * It returns a pointer to the collector.
     *
     * \return A pointer to collector node if exists, otherwise a \p NULL
     */
    ff_node* getCollector() { 
        if (collector == (ff_node*)gt) return NULL;
        return collector;
    }

protected:
    bool has_input_channel; // for accelerator
    bool prepared;
    bool collector_removed;
    int ondemand;          // if >0, emulates on-demand scheduling
    int in_buffer_entries;
    int out_buffer_entries;
    bool worker_cleanup;
    int max_nworkers;

    ff_node          *  emitter;
    ff_node          *  collector;

    lb_t             * lb;
    gt_t             * gt;
    svector<ff_node*>  workers;
    svector<ff_node*>  internalSupportNodes;
    bool               fixedsize;
};


/**
 *
 * \ingroup building_blocks
 *  
 *  \brief Ordered farm emitter
 *
 *  This class is defined in \ref farm.hpp
 */
class ofarm_lb: public ff_loadbalancer {
protected:
    /**
     * \brief Get selected worker
     *
     * Inspect which worker has been selected
     *
     * \return An integer value showing the worker.
     */
    inline size_t selectworker() { return victim; }
public:
    /**
     * \brief Constructor
     *
     * \param max_num_workers defines the maximum number of workers.
     */
    ofarm_lb(int max_num_workers):ff_loadbalancer(max_num_workers) {}

    /**
     * \internal
     * \brief Sets the target worker
     *
     * Overload to define you own scheduling policy
     *
     * \param v is the number of the worker.
     */
    void set_victim(size_t v) { victim=v;}
private:
    size_t victim;
};

/*!
 * \ingroup building_blocks
 *
 * \brief Ordered farm  Collector
 *
 * It defines an ordered farm
 *
 * This class is defined in \ref farm.hpp
 */
class ofarm_gt: public ff_gatherer {
protected:
    /**
     * \internal
     * \brief Selects worker
     *
     * It selects a worker
     *
     * \return An integet value showing he id of the worker
     */
    inline ssize_t selectworker() { return victim; }
public:
    /**
     * \brief Constructor
     *
     * \param max_num_workers defines the maximum number of workers.
     *
     */
    ofarm_gt(int max_num_workers):
        ff_gatherer(max_num_workers),dead(max_num_workers) {
        dead.resize(max_num_workers);
        revive();
    }

    /**
     * \internal
     * \brief Sets the victim
     *
     * It sets the victim with the given worker
     *
     * \return \p true if successful, otherwise \p false.
     */
    inline bool set_victim(size_t v) { 
        if (dead[v]) return false; 
        victim=v; 
        return true;
    }

    /**
     * \internal
     * \brief Sets the dead
     *
     * It sets an element in the dead vector to true, to make it dead.
     *
     * \param v is number in the dead vector.
     */
    inline void set_dead(ssize_t v)   { dead[v]=true;}

    /**
     * \internal
     *
     * It makes the element in the dead vector as false i.e. makes it alive.
     *
     */
    inline void revive() {
        for(size_t i=0;i<dead.size();++i) dead[i]=false;
    }
private:
    size_t victim;
    svector<bool> dead;
};

/*!
 *  \class ff_ofarm
 *  \ingroup core_patterns
 *
 *  \brief The ordered Farm skeleton.
 *
 *  This class is defined in \ref farm.hpp
 */

class ff_ofarm: public ff_farm<ofarm_lb, ofarm_gt> {
private:
    /**
     * \brief Ordered farm
     */
    class ofarmE: public ff_node {
        static bool ff_send_out_ofarmE(void * task,unsigned int retry,unsigned int ticks, void *obj) {
            return ((ofarmE *)obj)->ff_send_out(task, retry, ticks);
        }
    public:
        
        /**
         * \internal
         * \brief Emitter of ordered form
         *
         * It creates an ordered form
         *
         * \param lb is ordered farm load balancer
         *
         */
        ofarmE(ofarm_lb * const lb):
            nworkers(0),nextone(0), lb(lb),E_f(NULL) {}

        /**
         * \internal
         * \brief Sets number of workers
         *
         * It sets the number of workers.
         *
         * \param nw is the number of workers
         */
        void setnworkers(size_t nw) { nworkers=nw;}

        /**
         * \brief Set filtering policy for scheduling
         *
         */
        void setfilter(ff_node* f) { 
            E_f = f;
            if (f) f->registerCallback(ff_send_out_ofarmE, this);
        }

        /**
         * \brief \p svc_init method
         *
         * It defines the svc initialization method.
         *
         * \return 0 if not successful, otherwise the status of \p svc_init
         */
        int svc_init() {
            assert(nworkers>0);
            int ret = 0;
            if (E_f) ret = E_f->svc_init();
            nextone = 0;
            lb->set_victim(nextone);
            return ret;
        }  

        /**
         * \brief \p svc method
         *
         * It defines the \p svc method.
         *
         * \param task is a void pointer
         *
         * \return If it is \p FF_EOS then the status of svc method is returned
         * otherwise \p GO_ON
         *
         */
        void * svc(void * task) {
            if (E_f) task = E_f->svc(task);
            if (task == (void*)FF_EOS) return task;
            ff_send_out(task);
            nextone = (nextone+1) % nworkers;
            lb->set_victim(nextone);
            return GO_ON;
        }

        /**
         * \brief \p svc_end method
         *
         * It defines the \p svc_end method.
         */
        void svc_end() {
            if (E_f) E_f->svc_end();
        }
    private:
        size_t nworkers, nextone;
        ofarm_lb * lb;
        ff_node* E_f;
    };

    /**
     * \ingroup building_blocks
     *
     * \brief Ordered farm default Emitter
     */
    class ofarmC: public ff_node {
        static bool ff_send_out_ofarmC(void * task,unsigned int retry,unsigned int ticks, void *obj) {
            return ((ofarmC *)obj)->ff_send_out(task, retry, ticks);
        }
    public:
        /**
         * \brief Constructor
         *
         */
        ofarmC(ofarm_gt * const gt):
            nworkers(0),nextone(0), gt(gt),C_f(NULL) {}

        /**
         * \internal
         * \brief Sets worker
         *
         * It sets the number of workers
         *
         * \param nw is the number of workers
         */
        void setnworkers(size_t nw) { nworkers=nw;}

        /**
         * \internal
         *
         * \brief Sets filtering policy
         *
         * It sets the filter.
         *
         */
        void setfilter(ff_node* f) { 
            C_f = f;
            if (f) f->registerCallback(ff_send_out_ofarmC, this);
        }

        /**
         * \brief \p svc_init method
         *
         * It defines the \p svc_init method.
         *
         * \return 0 if successful, otherwise the status of \p set_victim
         */
        int svc_init() {
            assert(nworkers>0);
            int ret = 0;
            if (C_f) ret = C_f->svc_init();
            nextone=0;
            gt->revive();
            gt->set_victim(nextone);
            return ret;
        }

        /**
         * \brief \p svc method
         *
         * It defines the \p svc method.
         *
         * \return \p GO_ON is always returned.
         */
        void * svc(void * task) {
            if (C_f) task = C_f->svc(task);
            if (ff_node::get_out_buffer()) ff_send_out(task);
            do nextone = (nextone+1) % nworkers;
            while(!gt->set_victim(nextone));
            return GO_ON;
        }

        /**
         * \internal
         * \brief Notifies EOS
         *
         * \It notifies the EOS.
         *
         */
        void eosnotify(ssize_t id=-1) { 
            gt->set_dead(id);
            if (nextone == (size_t)id) {
                nextone= (nextone+1) % nworkers;
                gt->set_victim(nextone);
            }
        }

        /**
         * \brief \p svc_end method
         *
         * It defines the \p svc_end method.
         */
        void svc_end() {
            if (C_f) C_f->svc_end();
        }
    private:
        size_t nworkers, nextone;
        ofarm_gt * gt;
        ff_node* C_f;
    };
    
public:
    /**
     * \brief Constructor
     *
     * \param input_ch states the presence of input channel
     * \param in_buffer_entries defines the number of input entries
     * \param worker_cleanup states the cleaning of workers
     * \param max_num_workers defines the maximum number of workers in the
     * ordered farm
     * \param fixedsize states the status of fixed size buffer
     *
     */
    ff_ofarm(bool input_ch=false,
            int in_buffer_entries=DEF_IN_BUFF_ENTRIES, 
            int out_buffer_entries=DEF_OUT_BUFF_ENTRIES,
            bool worker_cleanup=false,
            int max_num_workers=DEF_MAX_NUM_WORKERS,
            bool fixedsize=false):  // NOTE: by default all the internal farm queues are unbounded !
        ff_farm<ofarm_lb,ofarm_gt>(input_ch,in_buffer_entries,out_buffer_entries,worker_cleanup,max_num_workers,fixedsize),E(NULL),C(NULL),E_f(NULL),C_f(NULL) {
        E = new ofarmE(this->getlb());
        C = new ofarmC(this->getgt());
        this->add_emitter(E);
        this->add_collector(C);
    }

    /**
     * \brief Destructor 
     *
     * It clean emitter and collector.
     */
    ~ff_ofarm() {
        if (E) delete E;
        if (C) delete C;
    }

    /**
     * \internal
     *  \brief Sets emitter
     *
     * It sets the emitter.
     *
     * \param f is the FastFlow node.
     */
    void setEmitterF  (ff_node* f) { E_f = f; }

    /**
     * \internal
     *  \brief Sets collector
     *
     * It sets the collecto
     *
     * \param f is the FastFlow node.
     */
    void setCollectorF(ff_node* f) { C_f = f; }
    
    /**
     * \brief run
     *
     */
    int run(bool skip_init=false) {
        E->setnworkers(this->getNWorkers());
        C->setnworkers(this->getNWorkers());
        E->setfilter(E_f);
        C->setfilter(C_f);
        return ff_farm<ofarm_lb,ofarm_gt>::run(skip_init);
    }
protected:
    ofarmE* E;
    ofarmC* C;
    ff_node* E_f;
    ff_node* C_f;
};


    /* ************************* Multi-Input node ************************* */

/*!
 * \ingroup building_blocks
 *
 * \brief Multiple input ff_node (the SPMC mediator)
 *
 * The ff_node with many input channels.
 *
 * This class is defined in \ref farm.hpp
 */

class ff_minode: public ff_node {
protected:

    /**
     * \brief Gets the number of input channels
     */
    inline int cardinality(BARRIER_T * const barrier)  { 
        gt->set_barrier(barrier);
        return 1;
    }

    /**
     * \brief Creates the input channels
     *
     * \return >=0 if successful, otherwise -1 is returned.
     */
    int create_input_buffer(int nentries, bool fixedsize=true) {      
        if (inputNodes.size()==0) {
            int r = ff_node::create_input_buffer(nentries, fixedsize);
            if (r!=0) return r;
            return 1;
        }
        return 0;
    }

    int  wait(/* timeout */) { 
        if (gt->wait()<0) return -1;
        return 0;
    }

public:
    /**
     * \brief Constructor
     */
    ff_minode(int max_num_workers=ff_farm<>::DEF_MAX_NUM_WORKERS):
        ff_node(), gt(new ff_gatherer(max_num_workers)) { ff_node::setMultiInput(); }

    /**
     * \brief Destructor 
     */
    virtual ~ff_minode() {
        if (gt) delete gt;
    }
    
    /**
     * \brief Assembly input channels
     *
     * Assembly input channelnames to ff_node channels
     */
    virtual inline int set_input(svector<ff_node *> & w) { 
        inputNodes += w;
        return 0; 
    }

    /**
     * \brief Assembly a input channel
     *
     * Assembly a input channelname to a ff_node channel
     */
    virtual inline int set_input(ff_node *node) { 
        inputNodes.push_back(node); 
        return 0;
    }

    virtual bool isMultiInput() const { return true;}

    virtual inline void get_out_nodes(svector<ff_node*>&w) {
        w.push_back(this);
    }

    /**
     * \brief Skip first pop
     *
     * Set up spontaneous start
     */
    inline void skipfirstpop(bool sk)   { ff_node::skipfirstpop(sk);}

    /**
     * \brief run
     *
     * \return 0 if successful, otherwise -1 is returned.
     *
     */
    int run(bool skip_init=false) {
        if (!gt) return -1;
        
        gt->set_filter(this);

        for(size_t i=0;i<inputNodes.size();++i)
            gt->register_worker(inputNodes[i]);
        if (ff_node::skipfirstpop()) gt->skipfirstpop();       
        if (gt->run()<0) {
            error("ff_minode, running gather module\n");
            return -1;
        }
        return 0;
    }
    int freeze_and_run(bool=false) { 
        gt->freeze();
        return run();
    }
    int run_then_freeze(ssize_t nw=-1) {
        if (gt->isfrozen()) {
            // true means that next time threads are frozen again
            gt->thaw(true, nw); 
            return 0;
        }
        gt->freeze();
        return run();
    }



    int wait_freezing() { return gt->wait_freezing(); }


    /**
     * \brief Gets the channel id from which the data has just been received
     *
     */
    ssize_t get_channel_id() const { return gt->get_channel_id();}

    /**
     * \internal
     * \brief Gets the gt
     *
     * It gets the internal gatherer.
     *
     * \return A pointer to the FastFlow gatherer.
     */
    inline ff_gatherer * getgt() const { return gt;}

#if defined(TRACE_FASTFLOW) 
    /**
     * \brief Prints the FastFlow trace
     *
     * It prints the trace of FastFlow.
     */
    inline void ffStats(std::ostream & out) { 
        gt->ffStats(out);
    }
#endif

private:
    svector<ff_node*> inputNodes;
    ff_gatherer* gt;
};

/*!
 *  \class ff_minode_t
 *  \ingroup building_blocks
 *
 *  \brief Typed multiple input ff_node (the SPMC mediator).
 *
 *  Key method is: \p svc (pure virtual).
 *
 *  This class is defined in \ref node.hpp
 */

template <typename T>
struct ff_minode_t: ff_minode {
    ff_minode_t():
        GO_ON((T*)FF_GO_ON),
        EOS((T*)FF_EOS),
        GO_OUT((T*)FF_GO_OUT),
        EOS_NOFREEZE((T*)FF_EOS_NOFREEZE) {}
    T *GO_ON, *EOS, *GO_OUT, *EOS_NOFREEZE;
    virtual ~ff_minode_t()  {}
    void *svc(void *task) { return svc(reinterpret_cast<T*>(task));};
    virtual T* svc(T*)=0;
};


    /* ************************* Multi-Ouput node ************************* */

/*!
 *  \ingroup building_blocks
 *
 * \brief Multiple output ff_node (the MPSC mediator)
 *
 * The ff_node with many output channels.
 *
 * This class is defined in \ref farm.hpp
 */

class ff_monode: public ff_node {
protected:
    /**
     * \brief Cardinatlity
     *
     * Defines the cardinatlity of the FastFlow node.
     *
     * \param barrier defines the barrier
     *
     * \return 1 is always returned.
     */
    inline int   cardinality(BARRIER_T * const barrier)  { 
        lb->set_barrier(barrier);
        return 1;
    }

    int  wait(/* timeout */) { 
        if (lb->waitlb()<0) return -1;
        return 0;
    }

public:
    /**
     * \brief Constructor
     *
     * \param max_num_workers defines the maximum number of workers
     *
     */
    ff_monode(int max_num_workers=ff_farm<>::DEF_MAX_NUM_WORKERS):
        ff_node(), lb(new ff_loadbalancer(max_num_workers)) {}

    /**
     * \brief Destructor 
     */
    virtual ~ff_monode() {
        if (lb) delete lb;
    }
    
    /**
     * \brief Assembly the output channels
     *
     * Attach output channelnames to ff_node channels
     */
    virtual inline int set_output(svector<ff_node *> & w) {
        for(size_t i=0;i<w.size();++i)
            outputNodes.push_back(w[i]);
        return 0; 
    }

    /**
     * \brief Assembly an output channels
     *
     * Attach a output channelname to ff_node channel
     */
    virtual inline int set_output(ff_node *node) { 
        outputNodes.push_back(node); 
        return 0;
    }

    virtual bool isMultiOutput() const { return true;}

    virtual inline void get_out_nodes(svector<ff_node*>&w) {
        w = outputNodes;
    }

    /**
     * \brief Skips the first pop
     *
     * Set up spontaneous start
     */
    inline void skipfirstpop(bool sk)   {
        if (sk) lb->skipfirstpop();
    }

    /**
     * \brief Sends one task to a specific node id.
     *
     */
    inline bool ff_send_out_to(void *task, int id) {
        return lb->ff_send_out_to(task,id);
    }
    
    /**
     * \brief run
     *
     * \param skip_init defines if the initilization should be skipped
     *
     * \return 0 if successful, otherwsie -1 is returned.
     */
    int run(bool skip_init=false) {
        if (!lb) return -1;
        lb->set_filter(this);

        for(size_t i=0;i<outputNodes.size();++i)
            lb->register_worker(outputNodes[i]);
        if (ff_node::skipfirstpop()) lb->skipfirstpop();       
        if (lb->runlb()<0) {
            error("ff_monode, running loadbalancer module\n");
            return -1;
        }
        return 0;
    }

    int freeze_and_run(bool=false) {
        lb->freeze();
        return run(true);
    }

    /**
     * \internal
     * \brief Gets the internal lb (Emitter)
     *
     * It gets the internal lb (Emitter)
     *
     * \return A pointer to the lb
     */
    inline ff_loadbalancer * getlb() const { return lb;}

#if defined(TRACE_FASTFLOW) 
    /*
     * \brief Prints the FastFlow trace
     *
     * It prints the trace of FastFlow.
     */
    inline void ffStats(std::ostream & out) { 
        lb->ffStats(out);
    }
#endif

protected:
    svector<ff_node*> outputNodes;
    ff_loadbalancer* lb;
};

/*!
 *  \class ff_minode_t
 *  \ingroup building_blocks
 *
 *  \brief Typed multiple output ff_node (the MPSC mediator).
 *
 *  Key method is: \p svc (pure virtual).
 *
 *  This class is defined in \ref node.hpp
 */

template <typename T>
struct ff_monode_t: ff_monode {
    ff_monode_t():
        GO_ON((T*)FF_GO_ON),
        EOS((T*)FF_EOS),
        GO_OUT((T*)FF_GO_OUT),
        EOS_NOFREEZE((T*)FF_EOS_NOFREEZE) {}
    T *GO_ON, *EOS, *GO_OUT, *EOS_NOFREEZE;
    virtual ~ff_monode_t()  {}
    void *svc(void *task) { return svc(reinterpret_cast<T*>(task));};
    virtual T* svc(T*)=0;
};



} // namespace ff

#endif /* FF_FARM_HPP */
