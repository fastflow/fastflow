/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 * \file pipeline.hpp
 * \ingroup core_patterns high_level_patterns
 *
 * \brief This file implements the pipeline skeleton, both in the high-level pattern
 * syntax (\ref ff::ff_pipe) and low-level syntax (\ref ff::ff_pipeline)
 *
 */
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

#ifndef FF_PIPELINE_HPP
#define FF_PIPELINE_HPP

#include <functional>
#include <ff/node.hpp>
#include <ff/svector.hpp>

#if defined( HAS_CXX11_VARIADIC_TEMPLATES )
#include <tuple>
#endif

namespace ff {

/* ---------------- basic high-level macros ----------------------- */

    /* FF_PIPEA creates an n-stage accelerator pipeline (max 16 stages) 
     * where channels have type 'type'. The functions called in each stage
     * must have the following signature:
     *   bool (*F)(type &)
     * If the function F returns false an EOS is produced in the 
     * output channel.
     *
     * Usage example:
     *
     *  struct task_t { .....};
     *
     *  bool F(task_t &task) { .....; return true;}
     *  bool G(task_t &task) { .....; return true;}
     *  bool H(task_t &task) { .....; return true;}
     * 
     *  FF_PIPEA(pipe1, task_t, F, G);     // 2-stage
     *  FF_PIPEA(pipe2, task_t, F, G, H);  // 3-stage
     *
     *  FF_PIPEARUN(pipe1); FF_PIPEARUN(pipe2);
     *  for(...) {
     *     if (even_task) FF_PIPEAOFFLOAD(pipe1, new task_t(...));
     *     else FF_PIPEAOFFLOAD(pipe2, new task_t(...));
     *  }
     *  FF_PIEAEND(pipe1);  FF_PIPEAEND(pipe2);
     *  FF_PIEAWAIT(pipe1); FF_PIPEAWAIT(pipe2);
     *
     */
#define FF_PIPEA(pipename, type, ...)                                   \
    ff_pipeline _FF_node_##pipename(true);                              \
    _FF_node_##pipename.cleanup_nodes();                                \
    {                                                                   \
      class _FF_pipe_stage: public ff_node {                            \
          typedef bool (*type_f)(type &);                               \
          type_f F;                                                     \
      public:                                                           \
       _FF_pipe_stage(type_f F):F(F) {}                                 \
       void* svc(void *task) {                                          \
          type* _ff_in = (type*)task;                                   \
          if (!F(*_ff_in)) return NULL;                                 \
          return task;                                                  \
       }                                                                \
      };                                                                \
      ff_pipeline *p = &_FF_node_##pipename;                            \
      FOREACHPIPE(p->add_stage, __VA_ARGS__);                           \
    }

#define FF_PIPEARUN(pipename)  _FF_node_##pipename.run()
#define FF_PIPEAWAIT(pipename) _FF_node_##pipename.wait()
#define FF_PIPEAOFFLOAD(pipename, task)  _FF_node_##pipename.offload(task);
#define FF_PIPEAEND(pipename)  _FF_node_##pipename.offload(EOS)
#define FF_PIPEAGETRESULTNB(pipename, result) _FF_node_##pipename.load_result_nb((void**)result)
#define FF_PIPEAGETRESULT(pipename, result) _FF_node_##pipename.load_result((void**)result)
/* ---------------------------------------------------------------- */




/**
 * \class ff_pipeline
 * \ingroup core_patterns
 *
 *  \brief The Pipeline skeleton (low-level syntax)
 *
 */
 
class ff_pipeline: public ff_node {
protected:
    inline int prepare() {
        // create input FFBUFFER
        int nstages=static_cast<int>(nodes_list.size());
        for(int i=1;i<nstages;++i) {
            if (nodes_list[i]->isMultiInput()) {
                if (nodes_list[i-1]->create_output_buffer(out_buffer_entries, 
                                                          fixedsize)<0)
                    return -1;
                svector<ff_node*> w(256);
                nodes_list[i-1]->get_out_nodes(w);
                if (w.size() == 0)
                    nodes_list[i]->set_input(nodes_list[i-1]);
                else
                    nodes_list[i]->set_input(w);
            } else {
                if (nodes_list[i]->create_input_buffer(in_buffer_entries, fixedsize)<0) {
                    error("PIPE, creating input buffer for node %d\n", i);
                    return -1;
                }
            }
        }
        
        // set output buffer
        for(int i=0;i<(nstages-1);++i) {
            if (nodes_list[i+1]->isMultiInput()) continue;
            if (nodes_list[i]->isMultiOutput()) {
                nodes_list[i]->set_output(nodes_list[i+1]);
                continue;
            }
            if (nodes_list[i]->set_output_buffer(nodes_list[i+1]->get_in_buffer())<0) {
                error("PIPE, setting output buffer to node %d\n", i);
                return -1;
            }
        }
        
        // Preparation of buffers for the accelerator
        int ret = 0;
        if (has_input_channel) {
            if (create_input_buffer(in_buffer_entries, fixedsize)<0) {
                error("PIPE, creating input buffer for the accelerator\n");
                ret=-1;
            } else {             
                if (get_out_buffer()) {
                    error("PIPE, output buffer already present for the accelerator\n");
                    ret=-1;
                } else {
                    if (create_output_buffer(out_buffer_entries,fixedsize)<0) {
                        error("PIPE, creating output buffer for the accelerator\n");
                        ret = -1;
                    }
                }
            }
        }
        prepared=true; 
        return ret;
    }


    int freeze_and_run(bool skip_init=false) {
        int nstages=static_cast<int>(nodes_list.size());
        if (!skip_init) {            
            // set the initial value for the barrier 
            if (!barrier)  barrier = new BARRIER_T;
            const int nthreads = cardinality(barrier);
            if (nthreads > MAX_NUM_THREADS) {
                error("PIPE, too much threads, increase MAX_NUM_THREADS !\n");
                return -1;
            }
            barrier->barrierSetup(nthreads);
        }
        if (!prepared) if (prepare()<0) return -1;
        int startid = (get_my_id()>0)?get_my_id():0;
        for(int i=0;i<nstages;++i) {
            nodes_list[i]->set_id(i+startid);
            if (nodes_list[i]->freeze_and_run(true)<0) {
                error("ERROR: PIPE, (freezing and) running stage %d\n", i);
                return -1;
            }
        }
        return 0;
    } 

public:
 
    enum { DEF_IN_BUFF_ENTRIES=512, DEF_OUT_BUFF_ENTRIES=(DEF_IN_BUFF_ENTRIES+128)};

    /**
     *  \brief Constructor
     *
     *  \param input_ch \p true set accelerator mode
     *  \param in_buffer_entries input queue length
     *  \param out_buffer_entries output queue length
     *  \param fixedsize \p true uses bound channels (SPSC queue)
     */
    ff_pipeline(bool input_ch=false,
                int in_buffer_entries=DEF_IN_BUFF_ENTRIES,
                int out_buffer_entries=DEF_OUT_BUFF_ENTRIES, 
                bool fixedsize=true):
        has_input_channel(input_ch),prepared(false),
        node_cleanup(false),fixedsize(fixedsize),
        in_buffer_entries(in_buffer_entries),
        out_buffer_entries(out_buffer_entries) {               
    }
    
    /**
     * \brief Destructor
     */
    ~ff_pipeline() {
        if (end_callback) end_callback(end_callback_param);
        if (barrier) delete barrier;
        if (node_cleanup) {
            while(nodes_list.size()>0) {
                ff_node *n = nodes_list.back();
                nodes_list.pop_back();
                delete n;
            }
        }
        for(size_t i=0;i<internalSupportNodes.size();++i) {
            delete internalSupportNodes.back();
            internalSupportNodes.pop_back();
        }
    }

    //  WARNING: if these methods are called after prepare (i.e. after having called
    //  run_and_wait_end/run_then_freeze/run/....) they have no effect.
     
    void setXNodeInputQueueLength(int sz) { in_buffer_entries = sz; }
    void setXNodeOutputQueueLength(int sz) { out_buffer_entries = sz;}

    /**
     *  \brief It adds a stage to the pipeline
     *
     *  \param s a ff_node (or derived, e.g. farm) object that is the stage to be added 
     *  to the pipeline
     */
    int add_stage(ff_node * s) {
        if (nodes_list.size()==0 && s->isMultiInput())
            ff_node::setMultiInput();
        nodes_list.push_back(s);        
        return 0;
    }

    inline void setMultiOutput() { 
        ff_node::setMultiOutput();        
    }


    /**
     * \brief Feedback channel (pattern modifier)
     * 
     * The last stage output stream will be connected to the first stage 
     * input stream in a cycle (feedback channel)
     */
    int wrap_around(bool multi_input=false) {
        if (nodes_list.size()<2) {
            error("PIPE, too few pipeline nodes\n");
            return -1;
        }

        fixedsize=false; // NOTE: forces unbounded size for the queues!
        int last = static_cast<int>(nodes_list.size())-1;
        if (nodes_list[0]->isMultiInput()) {
            if (nodes_list[last]->isMultiOutput()) {
                ff_node *t = new ff_buffernode(out_buffer_entries,fixedsize);
                if (!t) return -1;
                t->set_id(last); // NOTE: that's not the real node id !
                internalSupportNodes.push_back(t);
                nodes_list[0]->set_input(t);
                nodes_list[last]->set_output(t);
            } else {
                if (create_output_buffer(out_buffer_entries, fixedsize)<0)
                    return -1;
                svector<ff_node*> w(256);
                this->get_out_nodes(w);
                nodes_list[0]->set_input(w);
            }
            if (!multi_input) nodes_list[0]->skipfirstpop(true);

        } else {
            if (create_input_buffer(out_buffer_entries, fixedsize)<0)
                return -1;
            
            if (nodes_list[last]->isMultiOutput()) 
                nodes_list[last]->set_output(nodes_list[0]);
            else 
                if (set_output_buffer(get_in_buffer())<0)
                    return -1;            
            
            nodes_list[0]->skipfirstpop(true);
        }

        return 0;
    }

   
    inline void cleanup_nodes() { node_cleanup = true; }


    inline void get_out_nodes(svector<ff_node*>&w) {
        assert(nodes_list.size()>0);
        int last = static_cast<int>(nodes_list.size())-1;
        nodes_list[last]->get_out_nodes(w);
        if (w.size()==0) 
            w.push_back(nodes_list[last]);
    }


    /**
     * \brief Run the pipeline skeleton asynchronously
     * 
     * Run the pipeline, the method call return immediately. To be coupled with 
     * \ref ff_pipeline::wait()
     */
    int run(bool skip_init=false) {
        int nstages=static_cast<int>(nodes_list.size());

        if (!skip_init) {            
            // set the initial value for the barrier 
            if (!barrier)  barrier = new BARRIER_T;
            const int nthreads = cardinality(barrier);
            if (nthreads > MAX_NUM_THREADS) {
                error("PIPE, too much threads, increase MAX_NUM_THREADS !\n");
                return -1;
            }
            barrier->barrierSetup(nthreads);
        }
        if (!prepared) if (prepare()<0) return -1;

        int startid = (get_my_id()>0)?get_my_id():0;
        for(int i=0;i<nstages;++i) {
            nodes_list[i]->set_id(i+startid);
            if (nodes_list[i]->run(true)<0) {
                error("ERROR: PIPE, running stage %d\n", i);
                return -1;
            }
        }        
        return 0;
    }

    int dryrun() {  
        if (!prepared) 
            if (prepare()<0) return -1; 
        return 0;
    }

    /**
     * \relates ff_pipe
     * \brief run the pipeline, waits that all stages received the End-Of-Stream (EOS),
     * and destroy the pipeline run-time 
     * 
     * Blocking behaviour w.r.t. main thread to be clarified
     */
    int run_and_wait_end() {
        if (isfrozen()) {  // TODO 
            error("PIPE: Error: feature not yet supported\n");
            return -1;
        } 
        stop();
        if (run()<0) return -1;           
        if (wait()<0) return -1;
        return 0;
    }
    
    /**
     * \related ff_pipe
     * \brief run the pipeline, waits that all stages received the End-Of-Stream (EOS),
     * and suspend the pipeline run-time
     * 
     * Run-time threads are suspended by way of a distrubuted protocol. 
     * The same pipeline can be re-started by calling again run_then_freeze 
     */
    virtual int run_then_freeze(bool skip_init=false) {
        if (isfrozen()) {
            // true means that next time threads are frozen again
            thaw(true);
            return 0;
        }
        if (!prepared) if (prepare()<0) return -1;
        //freeze();
        //return run();
        /* freeze_and_run is required because in the pipeline 
         * there isn't no manager thread, which allows to freeze other 
         * threads before starting the computation
         */
        return freeze_and_run(skip_init);
    }
    
    /**
     * \brief wait for pipeline termination (all stages received EOS)
     */
    int wait(/* timeval */ ) {
        int ret=0;
        for(unsigned int i=0;i<nodes_list.size();++i)
            if (nodes_list[i]->wait()<0) {
                error("PIPE, waiting stage thread, id = %d\n",nodes_list[i]->get_my_id());
                ret = -1;
            } 
        
        return ret;
    }
    
    /**
     * \brief wait for pipeline to complete and suspend (all stages received EOS)
     * 
     * Should be coupled with ??? 
     */
    inline int wait_freezing(/* timeval */ ) {
        int ret=0;
        for(unsigned int i=0;i<nodes_list.size();++i)
            if (nodes_list[i]->wait_freezing()<0) {
                error("PIPE, waiting freezing of stage thread, id = %d\n",
                      nodes_list[i]->get_my_id());
                ret = -1;
            } 
        
        return ret;
    } 
    
   
    inline void stop() {
        for(unsigned int i=0;i<nodes_list.size();++i) nodes_list[i]->stop();
    }

  
    inline void freeze() {
        for(unsigned int i=0;i<nodes_list.size();++i) nodes_list[i]->freeze();
    }

    
    inline void thaw(bool _freeze=false) {
        for(unsigned int i=0;i<nodes_list.size();++i) nodes_list[i]->thaw(_freeze);
    }
    
    inline bool isfrozen() const { 
        int nstages=static_cast<int>(nodes_list.size());
        for(int i=0;i<nstages;++i) 
            if (!nodes_list[i]->isfrozen()) return false;
        return true;
    }

    /** 
     * \brief offload a task to the pipeline from the offloading thread (accelerator mode)
     * 
     * Offload a task onto a pipeline accelerator, tipically the offloading 
     * entity is the main thread (even if it can be used from any 
     * \ref ff_node::svc method)  
     *
     * \note to be used in accelerator mode only
     */
    inline bool offload(void * task,
                        unsigned int retry=((unsigned int)-1),
                        unsigned int ticks=ff_node::TICKS2WAIT) { 
        FFBUFFER * inbuffer = get_in_buffer();
        assert(inbuffer != NULL);
        // if (!has_input_channel) 
        //     error("PIPE: accelerator is not set, offload not available\n");
        // else
        //     error("PIPE: input buffer creation failed\n");
        for(unsigned int i=0;i<retry;++i) {
            if (inbuffer->push(task)) return true;
            ticks_wait(ticks);
        }     
        return false;
    }    
    
    /** 
     * \brief gets a result from a task to the pipeline from the main thread 
     * (accelator mode)
     * 
     * Total call: return when a result is available. To be used in accelerator mode only
     *
     * \param[out] task
     * \param retry number of attempts to get a result before failing 
     * (related to nonblocking get from channel - expert use only)
     * \param ticks number of clock cycles between successive attempts 
     * (related to nonblocking get from channel - expert use only)
     * \return \p true is a task is returned, \p false if End-Of-Stream (EOS)
     */
    inline bool load_result(void ** task, 
                            unsigned int retry=((unsigned int)-1),
                            unsigned int ticks=ff_node::TICKS2WAIT) {
        FFBUFFER * outbuffer = get_out_buffer();
        if (outbuffer) {
            for(unsigned int i=0;i<retry;++i) {
                if (outbuffer->pop(task)) {
                    if ((*task != (void *)FF_EOS)) return true;
                    else return false;
                }
                ticks_wait(ticks);
            }     
            return false;
        }
        
        if (!has_input_channel) 
            error("PIPE: accelerator is not set, offload not available");
        else
            error("PIPE: output buffer not created");
        return false;
    }

/** 
     * \brief try to get a result from a task to the pipeline from the main thread 
     * (accelator mode)
     * 
     * Partial call: can return no result. To be used in accelerator mode only
     *
     * \param[out] task
     * \return \p true is a task is returned (including EOS), 
     * \p false if no task is returned
     */
    inline bool load_result_nb(void ** task) {
        FFBUFFER * outbuffer = get_out_buffer();
        if (outbuffer) {
            if (outbuffer->pop(task)) return true;
            else return false;
        }
        
        if (!has_input_channel) 
            error("PIPE: accelerator is not set, offload not available");
        else
            error("PIPE: output buffer not created");
        return false;        
    }
    

    int cardinality(BARRIER_T * const barrier)  { 
        int card=0;
        for(unsigned int i=0;i<nodes_list.size();++i) 
            card += nodes_list[i]->cardinality(barrier);
        
        return card;
    }
    
    /* 
     * \brief Misure execution time (including init and finalise)
     *
     * \return pipeline execution time (including init and finalise)
     */
    double ffTime() {
        return diffmsec(nodes_list[nodes_list.size()-1]->getstoptime(),
                        nodes_list[0]->getstarttime());
    }
    
    /* 
     * \brief Misure execution time (excluding init and finalise)
     *
     * \return pipeline execution time (excluding runtime setup)
     */
    double ffwTime() {
        return diffmsec(nodes_list[nodes_list.size()-1]->getwstoptime(),
                        nodes_list[0]->getwstartime());
    }
    
#if defined(TRACE_FASTFLOW)
    void ffStats(std::ostream & out) { 
        out << "--- pipeline:\n";
        for(unsigned int i=0;i<nodes_list.size();++i)
            nodes_list[i]->ffStats(out);
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }
#endif
    
protected:
    
    void* svc(void * task) { return NULL; }
    
    int   svc_init() { return -1; };
    
    void  svc_end()  {}
    
    void  setAffinity(int) { 
        error("PIPE, setAffinity: cannot set affinity for the pipeline\n");
    }
    
    int   getCPUId() { return -1;}

    int create_input_buffer(int nentries, bool fixedsize) { 
        if (in) return -1;  

        if (nodes_list[0]->create_input_buffer(nentries, fixedsize)<0) {
            error("PIPE, creating input buffer for node 0\n");
            return -1;
        }
        if (!nodes_list[0]->isMultiInput()) 
            ff_node::set_input_buffer(nodes_list[0]->get_in_buffer());

        return 0;
    }
    
    int create_output_buffer(int nentries, bool fixedsize=false) {
        int last = static_cast<int>(nodes_list.size())-1;
        if (last<0) return -1;

        if (nodes_list[last]->create_output_buffer(nentries, fixedsize)<0) {
            error("PIPE, creating output buffer for node %d\n",last);
            return -1;
        }
        ff_node::set_output_buffer(nodes_list[last]->get_out_buffer());
        return 0;
    }

    int set_output_buffer(FFBUFFER * const o) {
        int last = static_cast<int>(nodes_list.size())-1;
        if (!last) return -1;

        if (nodes_list[last]->set_output_buffer(o)<0) {
            error("PIPE, setting output buffer for node %d\n",last);
            return -1;
        }
        return 0;
    }

    inline bool isMultiInput() const { 
        if (nodes_list.size()==0) return false;
        return nodes_list[0]->isMultiInput();
    }
    inline bool isMultiOutput() const {
        if (nodes_list.size()==0) return false;
        int last = static_cast<int>(nodes_list.size())-1;
        return nodes_list[last]->isMultiOutput();
    }
    inline int set_input(ff_node *node) { 
        return nodes_list[0]->set_input(node);
    }
    inline int set_input(svector<ff_node *> & w) { 
             return nodes_list[0]->set_input(w);
    }   
    inline int set_output(ff_node *node) {
        int last = static_cast<int>(nodes_list.size())-1;
        return nodes_list[last]->set_output(node);
    }

private:
    bool has_input_channel; // for accelerator
    bool prepared;
    bool node_cleanup;
    bool fixedsize;
    int in_buffer_entries;
    int out_buffer_entries;
    svector<ff_node *> nodes_list;
    svector<ff_node*>  internalSupportNodes;
};


    // ------------------------ high-level (simpler) pipeline ------------------

// generic ff_node stage. It is built around the function F ( F: T* -> T* )
// template<typename T>
// class Fstage: public ff_node {
// public:
//     typedef T*(*F_t)(T*);
//     Fstage(F_t F):F(F) {}
//     inline void* svc(void *t) {	 return F((T*)t); }    
// protected:
//     F_t F;
// };

#if defined( HAS_CXX11_VARIADIC_TEMPLATES )

    // NOTE: std::function can introduce a bit of extra overhead.
    // Think about on how to avoid functionals.
    template<typename T>
    class Fstage2: public ff_node {
    public:
        Fstage2(const std::function<T*(T*,ff_node*const)> &F):F(F) {}
        inline void* svc(void *t) {	 return F((T*)t, this); }
    protected:
        std::function<T*(T*,ff_node*const)> F;
    };

    /** 
     * \class ff_pipe
     * \ingroup high_level_patterns
     * 
     * \brief Pipeline pattern (high-level pattern syntax)
     *
     * Set up a parallel for pipeline pattern run-time support object. 
     * Run with \p run_and_wait_end or \p run_the_freeze. See related functions.
     *
     * \note Don't use to model a workflow of tasks, stages are nonblocking threads 
     * and
     * require one core per stage. If you need to model a workflow use \ref ff::ff_mdf
     *
     * \example pipe_basic.cpp
     */ 
    template<typename TaskType>
    class ff_pipe: public ff_pipeline {
    private:
        typedef TaskType*(*F_t)(TaskType*);
        
        template<std::size_t I = 1, typename FuncT, typename... Tp>
        inline typename std::enable_if<I == (sizeof...(Tp)-1), void>::type
        for_each(std::tuple<Tp...> & t, FuncT f) { f(std::get<I>(t)); } // last one
        
        template<std::size_t I = 1, typename FuncT, typename... Tp>
        inline typename std::enable_if<I < (sizeof...(Tp)-1), void>::type
        for_each(std::tuple<Tp...>& t, FuncT f) {
            f(std::get<I>(t));
            for_each<I + 1, FuncT, Tp...>(t, f);  // all but the first
        }
            
            
        inline void add2pipe(ff_node *node) { ff_pipeline::add_stage(node); }
        //    inline void add2pipe(F_t F) 
        //{   ff_pipeline::add_stage(new Fstage<TaskType>(F));  }
        inline void add2pipe(std::function<TaskType*(TaskType*,ff_node*const)> F) { ff_pipeline::add_stage(new Fstage2<TaskType>(F));  }
        
        struct add_to_pipe {
            ff_pipe *const P;
            add_to_pipe(ff_pipe *const P):P(P) {}
            template<typename T>
            void operator()(T t) const { P->add2pipe(t); }
        };
        
    public:
        /**
         * \brief Create a stand-alone pipeline (no input/output streams). Run with \p run_and_wait_end or \p run_the_freeze.
         *
         * Identifies an stream parallel construct in which stages are executed 
         * in parallel. 
         * It does require a stream of tasks, either external of created by the 
         * first stage.
         * \param args pipeline stages, i.e. a list f1,f2,... of functions 
         * with the following type
         * <tt> const std::function<T*(T*,ff_node*const)></tt>
         * 
         * Example: \ref pipe_basic.cpp
         */
        template<typename... Arguments>
        ff_pipe(Arguments...args) {
            std::tuple<Arguments...> t = std::make_tuple(args...);
            auto firstF = std::get<0>(t);
            add2pipe(firstF);
            for_each(t,add_to_pipe(this));
        }
        /**
         * \brief Create a pipeline (with input stream). Run with \p run_and_wait_end or \p run_the_freeze.
         *
         * Identifies an stream parallel construct in which stages are executed 
         * in parallel. 
         * It does require a stream of tasks, either external of created by the 
         * first stage.
         * \param input_ch \p true to enable first stage input stream
         * \param args pipeline stages, i.e. a list f1,f2,... of functions 
         * with the following type
         * <tt> const std::function<T*(T*,ff_node*const)></tt>
         *
         * Example: \ref pipe_basic.cpp
         */
        template<typename... Arguments>
        ff_pipe(bool input_ch, Arguments...args):ff_pipeline(input_ch) {
            std::tuple<Arguments...> t = std::make_tuple(args...);
            auto firstF = std::get<0>(t);
            add2pipe(firstF);
            for_each(t,add_to_pipe(this));
        }
        
        operator ff_node* () { return this;}
    };
#endif /* HAS_CXX11_VARIADIC_TEMPLATES */

    template<typename T>
    ff_node* toffnode(T* p) { return p;}
    template<typename T>
    ff_node* toffnode(T& p) { return (ff_node*)p;}
    
    // ---------------------------------------------------------------



#define CONCAT(x, y)  x##y
#define FFPIPE_1(apply, x, ...)  apply(new _FF_pipe_stage(x))
#define FFPIPE_2(apply, x, ...)  apply(new _FF_pipe_stage(x)); FFPIPE_1(apply,  __VA_ARGS__)
#define FFPIPE_3(apply, x, ...)  apply(new _FF_pipe_stage(x)); FFPIPE_2(apply, __VA_ARGS__)
#define FFPIPE_4(apply, x, ...)  apply(new _FF_pipe_stage(x)); FFPIPE_3(apply,  __VA_ARGS__)
#define FFPIPE_5(apply, x, ...)  apply(new _FF_pipe_stage(x)); FFPIPE_4(apply,  __VA_ARGS__)
#define FFPIPE_6(apply, x, ...)  apply(new _FF_pipe_stage(x)); FFPIPE_5(apply,  __VA_ARGS__)
#define FFPIPE_7(apply, x, ...)  apply(new _FF_pipe_stage(x)); FFPIPE_6(apply,  __VA_ARGS__)
#define FFPIPE_8(apply, x, ...)  apply(new _FF_pipe_stage(x)); FFPIPE_7(apply,  __VA_ARGS__)
#define FFPIPE_9(apply, x, ...)  apply(new _FF_pipe_stage(x)); FFPIPE_8(apply,  __VA_ARGS__)
#define FFPIPE_10(apply, x, ...) apply(new _FF_pipe_stage(x)); FFPIPE_9(apply,  __VA_ARGS__)
#define FFPIPE_11(apply, x, ...) apply(new _FF_pipe_stage(x)); FFPIPE_10(apply,  __VA_ARGS__)
#define FFPIPE_12(apply, x, ...) apply(new _FF_pipe_stage(x)); FFPIPE_11(apply,  __VA_ARGS__)
#define FFPIPE_13(apply, x, ...) apply(new _FF_pipe_stage(x)); FFPIPE_12(apply,  __VA_ARGS__)
#define FFPIPE_14(apply, x, ...) apply(new _FF_pipe_stage(x)); FFPIPE_13(apply,  __VA_ARGS__)
#define FFPIPE_15(apply, x, ...) apply(new _FF_pipe_stage(x)); FFPIPE_14(apply,  __VA_ARGS__)
#define FFPIPE_16(apply, x, ...) apply(new _FF_pipe_stage(x)); FFPIPE_15(apply,  __VA_ARGS__)

#define FFPIPE_NARG(...) FFPIPE_NARG_(__VA_ARGS__, FFPIPE_RSEQ_N())
#define FFPIPE_NARG_(...) FFPIPE_ARG_N(__VA_ARGS__) 
#define FFPIPE_ARG_N(_1,_2,_3,_4,_5,_6,_7,_8, _9,_10,_11,_12,_13,_14,_15,_16, N, ...) N
#define FFPIPE_RSEQ_N() 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
#define FFPIPE_(N, apply, x, ...) CONCAT(FFPIPE_, N)(apply, x, __VA_ARGS__)
#define FFCOUNT_(_0,_1,_2,_3,_4,_5,_6,_7,_8, _9,_10,_11,_12,_13,_14,_15,_16, N, ...) N
#define FFCOUNT(...) COUNT_(X,##__VA_ARGS__, 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)

#define FOREACHPIPE(apply, x, ...) FFPIPE_(FFPIPE_NARG(x, __VA_ARGS__), apply, x, __VA_ARGS__)

} // namespace ff

#endif /* FF_PIPELINE_HPP */
