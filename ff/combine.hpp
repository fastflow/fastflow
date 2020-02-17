/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file combine.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow composition building block
 *
 * @detail FastFlow basic contanier for a shared-memory parallel activity 
 *
 */

#ifndef FF_COMBINE_HPP
#define FF_COMBINE_HPP

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

/*
 *   Author: Massimo Torquati
 *      
 */

#include <ff/node.hpp>
#include <ff/multinode.hpp>
#include <ff/pipeline.hpp>
#include <ff/ordering_policies.hpp>
#include <ff/farm.hpp>

namespace ff {


// forward declaration    
class ff_comb;
static const ff_pipeline combine_ofarm_farm(ff_farm& farm1, ff_farm& farm2);
template<typename T1, typename T2>    
static const ff_comb combine_nodes(T1& n1, T2& n2);
template<typename T1, typename T2>    
static std::unique_ptr<ff_node> unique_combine_nodes(T1& n1, T2& n2);

    
class ff_comb: public ff_minode {
    //
    // NOTE: the ff_comb appears either as a standard ff_node or as ff_minode depending on
    //       whether the first node is a standard node or a multi-input node.
    //

    template<typename T1, typename T2>    
    friend const ff_comb combine_nodes(T1& n1, T2& n2);

    template<typename T1, typename T2>    
    friend std::unique_ptr<ff_node> unique_combine_nodes(T1& n1, T2& n2);

    friend class ff_loadbalancer;
    friend class ff_gatherer;
    friend class ff_farm;
    friend class ff_a2a;
    
    // used if the last stage has no output channel
    static bool devnull(void*,unsigned long, unsigned long, void*) {return true;}

private:
    ff_node* getFirst() const {
        if (comp_nodes[0]->isComp())
            return ((ff_comb*)comp_nodes[0])->getFirst();
        return comp_nodes[0];
    }
    ff_node* getLast() const {
        if (comp_nodes[1]->isComp())
            return ((ff_comb*)comp_nodes[1])->getLast();
        return comp_nodes[1];
    }
    
    void registerAllGatherCallback(int (*cb)(void *,void **, void*), void * arg) {
        assert(isMultiInput());
        // NOTE: the gt of the first node will be replaced by the ff_comb gt.
        ff_minode::getgt()->registerAllGatherCallback(cb,arg);
    }

public:
    template<typename T1, typename T2>
    ff_comb(T1& n1, T2& n2) {
        add_node(n1,n2);
    }
    ff_comb(ff_node* n1, ff_node* n2, bool first_cleanup=false, bool second_cleanup=false){
        if (!n1 || !n2) {
            error("COMBINE, passing null pointer to constructor\n");
            return;
        }
        add_node(n1,n2);
        if (first_cleanup) {
            cleanup_stages.push_back(n1);
        }
        if (second_cleanup) {
            cleanup_stages.push_back(n2);
        }
    }

    ff_comb(const ff_comb& c) {
        for (auto s: c.comp_nodes) {
            if (s->isComp()) {
                comp_nodes.push_back(new ff_comb(*(ff_comb*)s));
                assert(comp_nodes.back());
                cleanup_stages.push_back(comp_nodes.back());                
            } else {
                comp_nodes.push_back(s);
            }
        }
        // this is a dirty part, we modify a const object.....
        ff_comb *dirty= const_cast<ff_comb*>(&c);
        for (size_t i=0;i<dirty->cleanup_stages.size();++i) {
            cleanup_stages.push_back(dirty->cleanup_stages[i]);
            dirty->cleanup_stages[i]=nullptr;
        }
    }
    
    virtual ~ff_comb() {
        for (auto s: cleanup_stages) {
            if (s) delete s;
        }
    }

    int run(bool skip_init=false) {
        if (!skip_init) {
            if (getFirst()->get_in_buffer() == nullptr)
                getFirst()->skipfirstpop(true);
        }

        if (!prepared) if (prepare()<0) return -1;

        // set blocking mode for the last node of the composition
        getLast()->blocking_mode(blocking_in);      
        if (comp_nodes[0]->isMultiInput()) {
            svector<ff_node*> w(1);
            getFirst()->get_in_nodes(w);
            if (w.size() == 0)  getFirst()->skipfirstpop(true);
            return ff_minode::run();
        }

        if (ff_node::run(true)<0) return -1;
        return 0;
    }

     int  wait() {
         if (comp_nodes[0]->isMultiInput())
             return ff_minode::wait();
         if (ff_node::wait()<0) return -1;
         return 0;
    }

    int run_and_wait_end() {
        if (isfrozen()) {  // TODO 
            error("COMB: Error: FEATURE NOT YET SUPPORTED\n");
            return -1;
        } 
        stop();
        if (run()<0) return -1;           
        if (wait()<0) return -1;
        return 0;
    }
    // NOTE: it is multi-input only if the first node is multi-input
    bool isMultiInput() const {
        if (getFirst()->isMultiInput()) return true;
        return false; //comp_multi_input;
    }
    // NOTE: it is multi-output only if the last node is multi-output
    bool isMultiOutput() const {
        if (getLast()->isMultiOutput()) return true;
        return false; //comp_multi_output;
    }        
    inline bool isComp() const        { return true; }

    double ffTime() {
        return diffmsec(getstoptime(),getstarttime());
    }
    double ffwTime() {
        return diffmsec(getwstoptime(),getwstartime());
    }

#if defined(TRACE_FASTFLOW)
    void ffStats(std::ostream & out) { 
        out << "--- Comp:\n";
        if (comp_nodes[0]->isMultiInput()) {
            ff_minode::ffStats(out);
        } else
            ff_node::ffStats(out);
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }
#endif
    
protected:
    ff_comb():ff_minode() {}

    template<typename T1, typename T2>
    inline bool check(T1* n1, T2* n2) {
        if (n1->isFarm() || n1->isAll2All() || n1->isPipe() ||
            n2->isFarm() || n2->isAll2All() || n2->isPipe()) {
            error("COMBINE, input nodes cannot be farm, all-2-all or pipeline building-blocks\n");
            return false;
        }        
        return true;
    }
    template<typename T1, typename T2>
    inline bool check(T1& n1, T2& n2) {
        return check(&n1, &n2);
    }
    void add_node(ff_node* n1, ff_node* n2) {
        if (!check(n1, n2)) return;
        n1->registerCallback(n2->ff_send_out_comp, n2);
        comp_nodes.push_back(n1);
        comp_nodes.push_back(n2);
    }
    template<typename T1>
    void add_node(const T1& n1, ff_node* n2) {
        T1 *node1 = new T1(n1);
        assert(node1);
        if (!check(node1, n2)) return;
        cleanup_stages.push_back(node1);
        comp_nodes.push_back(node1);
        comp_nodes.push_back(n2);
    }
    template<typename T1, typename T2>
    void add_node(T1& n1, T2& n2) {
        if (!check(&n1, &n2)) return;
        n1.registerCallback(n2.ff_send_out_comp, &n2);
        comp_nodes.push_back(&n1);
        comp_nodes.push_back(&n2);
    }
    template<typename T1, typename T2>
    void add_node(const T1& n1, const T2& n2) {
        T1 *node1 = new T1(n1);
        T2 *node2 = new T2(n2);
        assert(node1 && node2);
        cleanup_stages.push_back(node1);
        cleanup_stages.push_back(node2);
        add_node(*node1, *node2);
    }
    template<typename T1, typename T2>
    void add_node(T1& n1, const T2& n2) {
        T2 *node2 = new T2(n2);
        assert(node2);
        cleanup_stages.push_back(node2);
        add_node(n1, *node2);
    }
    template<typename T1, typename T2>
    void add_node(const T1& n1, T2& n2) {
        T1 *node1 = new T1(n1);
        assert(node1);
        cleanup_stages.push_back(node1);
        add_node(*node1, n2);
    }

    void skipfirstpop(bool sk)   {
        getFirst()->skipfirstpop(sk);
        ff_node::skipfirstpop(sk);
    }

    bool  put(void * ptr) { 
        return ff_node::put(ptr);
    }
       
    void registerCallback(bool (*cb)(void *,unsigned long,unsigned long,void *), void * arg) {
        comp_nodes[1]->registerCallback(cb,arg);
    }
    
    void connectCallback() {
        if (comp_nodes[0]->isComp())
            ((ff_comb*)comp_nodes[0])->connectCallback();
        if (comp_nodes[1]->isComp())
            ((ff_comb*)comp_nodes[1])->connectCallback();
        
        svector<ff_node*> w1(1);
        svector<ff_node*> w2(1);
        comp_nodes[0]->get_out_nodes(w1);
        comp_nodes[1]->get_in_nodes(w2);
        if (w1.size() == 0 && w2.size() == 0) return;
        if (w1.size()>1 || w2.size()>1) {
            error("COMP, connecting callbacks\n");
            return;
        }
        ff_node *n1 = (w1.size() == 0)? comp_nodes[0]:w1[0];
        ff_node *n2 = (w2.size() == 0)? comp_nodes[1]:w2[0];
        n1->registerCallback(n2->ff_send_out_comp, n2);
    }

    int dryrun() {
        if (prepared) return 0;
        if (comp_nodes[0]->dryrun()<0) return -1;
        if (comp_nodes[1]->dryrun()<0) return -1;
        return 0;
    }
    int prepare() {
        if (prepared) return 0;
        connectCallback();
        dryrun();

        // checking if the first node is a multi-input node
        ff_node *n1 = getFirst();
        if (n1->isMultiInput()) {
            // here we substitute the gt
            ((ff_minode*)n1)->setgt(ff_minode::getgt());

            if (ff_minode::prepare()<0) return -1;
        }

        // registering a special callback if the last stage does
        // not have an output channel
        ff_node *n2 = getLast();
        if (n2->isMultiOutput()) {
            svector<ff_node*> w(1);
            n2->get_out_nodes(w);
            if (w.size()==0) 
                n2->registerCallback(devnull, nullptr);   // devnull callback
            else {
                // NOTE: if the last stage of a composition is a multi-output node
                // we must be sure that the callback is not set.
                // see comments in lb.hpp (dryrun) and in multinode.hpp (ff_monode::ff_send_out_to)
                n2->registerCallback(nullptr, nullptr); // resetting callback
            }
        } else 
            if (n2->get_out_buffer() == nullptr)
            n2->registerCallback(devnull, nullptr);   // devnull callback

        
        prepared = true;
        return 0;
    }
    void set_multiinput() {
        // see farm.hpp
        // when the composition is passed as filter of a farm collector (which is a multi-input)
        // the filter is seen as multi-input because we want to avoid to call eosnotify 
        // too many times (see ff_comb::eosnotify)
        if (comp_nodes[0]->isComp())
            return comp_nodes[0]->set_multiinput();
        comp_multi_input=true;
    }
    void set_multioutput() {
        if (comp_nodes[1]->isComp())
            return comp_nodes[1]->set_multioutput();
        comp_multi_output=true;
    }

    void set_neos(ssize_t n) {
        getFirst()->set_neos(n);
    }
    
    inline int cardinality(BARRIER_T * const barrier)  { 
        ff_node::set_barrier(barrier);
        return ff_minode::cardinality(barrier);
    }
    
    virtual void set_id(ssize_t id) {
        myid = id;
        if (comp_nodes.size()) {
            for(size_t j=0;j<comp_nodes.size(); ++j) {
                comp_nodes[j]->set_id(myid);
            }
        }
    }

    int svc_init() {
        neos=0;
        for(size_t j=0;j<comp_nodes.size(); ++j) {
            int r;
            if ((r=comp_nodes[j]->svc_init())<0) return r; 
        }
        return 0;
    }

    inline void* svc_comp_node1(void* task, void*r) {
        void* ret = r, *r2;
        if (comp_nodes[1]->isComp())
            ret = comp_nodes[1]->svc(task);
        else {
            svector<void*> &localdata1 = comp_nodes[1]->get_local_data();
            size_t n = localdata1.size();
            for(size_t k=0;k<n;++k) {
                void *t = localdata1[k];
                if (t == FF_EOS) {
                    comp_nodes[1]->eosnotify();
                    /*ret = FF_GO_OUT;*/ // NOTE: we cannot exit if the previous comp node does not exit
                    if (comp_nodes[1]->isMultiOutput())
                        comp_nodes[1]->propagateEOS(FF_EOS);         
                    else {
                        if (comp_multi_output) ret = FF_EOS;
                        else comp_nodes[1]->ff_send_out(FF_EOS); // if no output is present it calls devnull
                    }
                    break;
                }
                r2= comp_nodes[1]->svc(t);
                if (r2 == FF_GO_ON || r2 == FF_GO_OUT || r2 == FF_EOS_NOFREEZE) continue;
                if (r2 == FF_EOS) {
                    ret = FF_GO_OUT;
                    if (comp_nodes[1]->isMultiOutput())
                        comp_nodes[1]->propagateEOS(FF_EOS);
                    else {
                        if (comp_multi_output) ret = FF_EOS;
                        comp_nodes[1]->ff_send_out(FF_EOS); // if no output is present it calls devnull
                    }
                    break;
                }
                comp_nodes[1]->ff_send_out(r2);  // if no output is present it calls devnull
            }
            localdata1.clear();
        }
        return ret;
    }
    
    // main service function
    void *svc(void *task) {
        void *ret = FF_GO_ON;
        void *r1;
        
        if (comp_nodes[0]->isComp())
            ret = comp_nodes[0]->svc(task);
        else {
            svector<void*> &localdata1 = comp_nodes[0]->get_local_data();
            size_t n = localdata1.size();
            if (n>0) {
                for(size_t k=0;k<n;++k) {
                    void *t = localdata1[k];
                    if (t == FF_EOS) {
                        comp_nodes[0]->eosnotify();
                        ret = FF_GO_OUT;
                        comp_nodes[1]->push_comp_local(t);
                        break;
                    }
                    r1= comp_nodes[0]->svc(t);
                    if (r1 == FF_GO_ON || r1 == FF_GO_OUT || r1 == FF_EOS_NOFREEZE) continue;
                    comp_nodes[1]->push_comp_local(r1);
                    if (r1 == FF_EOS) {
                        ret = FF_GO_OUT;
                        break;
                    }
                }
                localdata1.clear();
            } else {
                if (task || comp_nodes[0]->skipfirstpop()) {
                    r1= comp_nodes[0]->svc(task);
                    if (!(r1 == FF_GO_ON || r1 == FF_GO_OUT || r1 == FF_EOS_NOFREEZE))
                        comp_nodes[1]->push_comp_local(r1);
                    if (r1 == FF_EOS) ret=FF_GO_OUT;
                }
            }
        }
        ret = svc_comp_node1(nullptr, ret);        
        return ret;
    }

    void svc_end() {
        for(size_t j=0;j<comp_nodes.size(); ++j) {
            comp_nodes[j]->svc_end();
        }
    }

    bool push_comp_local(void *task) {        
        return comp_nodes[0]->push_comp_local(task);
    }
    
    int set_output(const svector<ff_node *> & w) {
        return comp_nodes[1]->set_output(w);
    }
    int set_output(ff_node *n) {
        return comp_nodes[1]->set_output(n);
    }
    int set_output_feedback(ff_node *n) {
        return comp_nodes[1]->set_output_feedback(n);
    }
    int set_input(const svector<ff_node *> & w) {

        //if (comp_nodes[0]->isComp()) return comp_nodes[0]->set_input(w); // WARNING: <--------

        //assert(comp_nodes[0]->isMultiInput());
        if (comp_nodes[0]->set_input(w)<0) return -1;
        // if the first node of the comp is a multi-input node
        // we have to set the input of the current ff_minode that
        // is implementing the composition
        return ff_minode::set_input(w);        
    }
    int set_input(ff_node *n) {
        //if (comp_nodes[0]->isComp()) return comp_nodes[0]->set_input(n); // WARNING: <-------


        //assert(comp_nodes[0]->isMultiInput());
        if (comp_nodes[0]->set_input(n)<0) return -1;
        // if the first node of the comp is a multi-input node
        // we have to set the input of the current ff_minode that
        // is implementing the composition
        return ff_minode::set_input(n);        
    }
    int set_input_feedback(ff_node *n) {
        //if (comp_nodes[0]->isComp()) return comp_nodes[0]->set_input_feedback(n); // WARNING: <-----

        //assert(comp_nodes[0]->isMultiInput());
        if (comp_nodes[0]->set_input_feedback(n)<0) return -1;
        // if the first node of the comp is a multi-input node
        // we have to set the input of the current ff_minode that
        // is implementing the composition
        return ff_minode::set_input_feedback(n);        
    }

    void blocking_mode(bool blk=true) {
        blocking_in=blocking_out=blk;
        ff_node *n = getLast();
        if (n) n->blocking_mode(blocking_in);
    }

    void set_scheduling_ondemand(const int inbufferentries=1) {
        if (!isMultiOutput()) return;
        ff_node* n= getLast();
        assert(n->isMultiOutput());
        n->set_scheduling_ondemand(inbufferentries);
    }
    int ondemand_buffer() const {
        if (!isMultiOutput()) return 0;
        ff_node* n= getLast();
        assert(n->isMultiOutput());
        return n->ondemand_buffer();
    }
   
    void eosnotify(ssize_t id=-1) {
        comp_nodes[0]->eosnotify();
        
        // the eosnotify might produce some data in output 
        void *ret = svc_comp_node1(nullptr, GO_ON);
        
        ++neos;
        // if the first node is multi-input or is a comp passed as filter to a farm collector,
        // then we have to call eosnotify only if we have received all EOSs
        if (comp_nodes[0]->isMultiInput() || comp_multi_input) {
            const ssize_t n=getFirst()->get_neos();
            if ((neos >= n) &&
                (ret != FF_GO_OUT) /*this means that svc_comp_node1 has already called eosnotify */
                )
                comp_nodes[1]->eosnotify(id);                
            return;
        }
        if (ret != FF_GO_OUT)
            comp_nodes[1]->eosnotify(id);
    }

    void propagateEOS(void *task=FF_EOS) {
        if (comp_nodes[1]->isComp()) {
            comp_nodes[1]->propagateEOS(task);
            return;
        }
        
        if (comp_nodes[1]->isMultiOutput() ||
            comp_multi_output)               
            comp_nodes[1]->propagateEOS(task);
        else
            comp_nodes[1]->ff_send_out(task);
    }

    void get_out_nodes(svector<ff_node*>&w) {
        size_t len=w.size();
        comp_nodes[1]->get_out_nodes(w);
        if (len == w.size() && !comp_nodes[1]->isComp())
            w.push_back(comp_nodes[1]);
    }
    void get_in_nodes(svector<ff_node*>&w) {
        size_t len=w.size();
        comp_nodes[0]->get_in_nodes(w);
        if (len == w.size() && !comp_nodes[0]->isComp())
            w.push_back(comp_nodes[0]);
    }
    
    void get_in_nodes_feedback(svector<ff_node*>&w) {
        comp_nodes[0]->get_in_nodes_feedback(w);
    }

    int create_input_buffer(int nentries, bool fixedsize=FF_FIXED_SIZE) {
        if (isMultiInput()) {
            int r= ff_minode::create_input_buffer(nentries,fixedsize);
            if (r<0) return r;
            svector<ff_node*> w(1);
            ff_minode::get_in_nodes(w);
            assert(w.size()==1);
            r=ff_node::set_input_buffer(w[0]->get_in_buffer());
            return r;
        }
        return ff_node::create_input_buffer(nentries,fixedsize);
    }
    int create_output_buffer(int nentries, bool fixedsize=FF_FIXED_SIZE) {
        return comp_nodes[1]->create_output_buffer(nentries,fixedsize);
    }
    FFBUFFER * get_in_buffer() const {
        //if (getFirst()->isMultiInput()) return nullptr;
        return ff_node::get_in_buffer();
    }

    int set_output_buffer(FFBUFFER * const o) {
        return comp_nodes[1]->set_output_buffer(o);
    }

    // a composition can be passed as filter to a farm emitter  
    void setlb(ff_loadbalancer *elb, bool cleanup=false) {
        comp_nodes[1]->setlb(elb, cleanup);
    }
    // a composition can be passed as filter to a farm collector
    void setgt(ff_gatherer *egt, bool cleanup=false) {
        comp_nodes[0]->setgt(egt, cleanup);
        ff_minode::setgt(egt, cleanup);
    }

    // consumer
    bool init_input_blocking(pthread_mutex_t   *&m,
                             pthread_cond_t    *&c,
                             bool feedback=true) {
        ff_node *n = getFirst();
        if (n->isMultiInput()) {
            // inits local gt, which is used for gathering tasks....
            bool r= ff_minode::init_input_blocking(m,c);
            if (!r) return false;
            // ... then, sets all p_cons_* on all input channels
            svector<ff_node*> w(1);
            n->get_in_nodes(w);
            n->get_in_nodes_feedback(w);
            for(size_t i=0;i<w.size(); ++i) 
                w[i]->set_output_blocking(m,c);
            return true;
        }
        return ff_node::init_input_blocking(m,c);
    }
    // producer
    bool init_output_blocking(pthread_mutex_t   *&m,
                              pthread_cond_t    *&c,
                              bool feedback=true) {
        return comp_nodes[1]->init_output_blocking(m,c);
    }
    void set_output_blocking(pthread_mutex_t   *&m,
                             pthread_cond_t    *&c) {
        comp_nodes[1]->set_output_blocking(m,c);
    }

    // the following calls are needed because a composition
    // uses as output channel(s) the one(s) of the second node.
    // these functions should not be called if the node is multi-output
    inline bool  get(void **ptr)                 { return comp_nodes[1]->get(ptr);}
    inline pthread_mutex_t   &get_prod_m()       { return comp_nodes[1]->get_prod_m();}
    inline pthread_cond_t    &get_prod_c()       { return comp_nodes[1]->get_prod_c();}

    inline pthread_mutex_t   &get_cons_m()  {
        ff_node *n = getFirst();
        if (n->isMultiInput()) return ff_minode::get_cons_m();
        return ff_node::get_cons_m();
    }
    inline pthread_cond_t    &get_cons_c()  {
        ff_node *n = getFirst();
        if (n->isMultiInput()) return ff_minode::get_cons_c();
        return ff_node::get_cons_c();
    }
    
    FFBUFFER *get_out_buffer() const {
        if (getLast()->isMultiOutput()) return nullptr;
        return comp_nodes[1]->get_out_buffer();
    }
    inline bool ff_send_out(void * task, 
                            unsigned long retry=((unsigned long)-1),
                            unsigned long ticks=(ff_node::TICKS2WAIT)) { 
        return comp_nodes[1]->ff_send_out(task,retry,ticks);
    }

    inline bool ff_send_out_to(void * task,int, unsigned long retry=((unsigned long)-1),
                               unsigned long ticks=(ff_node::TICKS2WAIT)) { 
        return comp_nodes[1]->ff_send_out(task,retry,ticks);
    }

    const struct timeval getstarttime() const {
        if (comp_nodes[0]->isMultiInput()) return ff_minode::getstarttime();
        return ff_node::getstarttime();
    }
    const struct timeval getstoptime()  const {
        if (comp_nodes[0]->isMultiInput()) return ff_minode::getstoptime();
        return ff_node::getstoptime();
    }
    const struct timeval getwstartime() const {
        if (comp_nodes[0]->isMultiInput()) return ff_minode::getwstartime();
        return ff_node::getwstartime();

    }
    const struct timeval getwstoptime() const {
        if (comp_nodes[0]->isMultiInput()) return ff_minode::getwstoptime();
        return ff_node::getwstoptime();
    }
    
private:
    svector<ff_node*> comp_nodes;
    svector<ff_node*> cleanup_stages;   
    bool comp_multi_input = false;
    bool comp_multi_output= false;
    ssize_t neos=0;
};


/* *************************************************************************** *
 *                                                                             *
 *                             helper functions                                *
 *                                                                             *
 * *************************************************************************** */
    
/**
 *  combines either basic nodes or ff_comb(s)
 *
 */
template<typename T1, typename T2>    
static inline const ff_comb combine_nodes(T1& n1, T2& n2) {
    ff_comb comp;
    comp.add_node(n1,n2);
    return comp;
}
    
/**
 *  combines either basic nodes or ff_comb(s) and returns a unique_ptr
 *  useful to add ff_comb as farm's workers
 */    
template<typename T1, typename T2>    
static inline std::unique_ptr<ff_node> unique_combine_nodes(T1& n1, T2& n2) {
    ff_comb *c = new ff_comb;
    assert(c);
    std::unique_ptr<ff_node> comp(c);
    if (!c->check(n1,n2)) return comp;
    c->add_node(n1,n2);
    return comp;
}

/**
 *  combines two stages returning a pipeline:
 *   - node1 and node2 standard nodes (or ff_comb)   --> pipeline(node1, node2)
 *   - node1 standard node and node2 is a farm       --> pipeline(node2)  (node1 is merged with node2's emitter)
 *   - node1 is a farm and node2 is a standard node  --> pipeline(node1)  (node2 is merged with node1's collector)
 *   - node1 and node2 are both farms                --> pipeline(node1, node2)  (collector is merged with emitter -- see case4.2 of combine_farms)
 *     (NOTE: if node1 is an ordered farm, then its collector is not removed)
 */   
static inline const ff_pipeline combine_nodes_in_pipeline(ff_node& node1, ff_node& node2, bool cleanup1=false, bool cleanup2=false) {
    if (node1.isAll2All() || node2.isAll2All()) {
        error("combine_nodes_in_pipeline, cannot be used if one of the nodes is A2A\n");
        return ff_pipeline();
    }
    if (node1.isOFarm()) {
        if (node2.isFarm()) {
            ff_farm *farm1 = reinterpret_cast<ff_farm*>(&node1);
            ff_farm *farm2 = reinterpret_cast<ff_farm*>(&node2);
            if (cleanup1) farm1->cleanup_all();
            if (cleanup2) farm2->cleanup_all();
            return combine_ofarm_farm(*farm1, *farm2);
        }
        error("combine_nodes_in_pipeline, FEATURE NOT YET SUPPORTED (node1 ordered farm and node1 standard or combine node\n");
        abort(); // TODO
        return ff_pipeline();
    }
    if (!node1.isFarm() && !node2.isFarm()) { // two sequential nodes
        ff_pipeline pipe;
        pipe.add_stage(&node1, cleanup1);
        pipe.add_stage(&node2, cleanup2);
        return pipe;
    } else if (!node1.isFarm() && node2.isFarm()) { // seq with farm's emitter
        ff_pipeline pipe;
        ff_farm* farm = reinterpret_cast<ff_farm*>(&node2);
        ff_node *e = farm->getEmitter();
        if (!e) farm->add_emitter(&node1);
        else {
            ff_comb *p = new ff_comb(&node1, e, cleanup1, farm->isset_cleanup_emitter());
            if (farm->isset_cleanup_emitter()) farm->cleanup_emitter(false);
            farm->change_emitter(p, true);
        }
        pipe.add_stage(farm, cleanup2);
        return pipe;
    } else if (node1.isFarm() && !node2.isFarm()) { // first farm and seq
        ff_pipeline pipe;
        ff_farm* farm = reinterpret_cast<ff_farm*>(&node1);
        ff_node *c = farm->getCollector();
        if (!c)  farm->add_collector(&node2, cleanup2);
        else {
            ff_comb *p = new ff_comb(c, &node2, farm->isset_cleanup_collector(), cleanup2);
            if (farm->isset_cleanup_collector()) farm->cleanup_collector(false);
            farm->remove_collector();
            farm->add_collector(p, true);
        }
        pipe.add_stage(farm, cleanup1);
        return pipe;	
    }
    assert(node1.isFarm() && node2.isFarm());   
    ff_farm* farm1 = reinterpret_cast<ff_farm*>(&node1);
    ff_farm* farm2 = reinterpret_cast<ff_farm*>(&node2);
    
    ff_node *e = farm2->getEmitter();
    ff_node *c = farm1->getCollector();
    ff_pipeline pipe;
    if (c) {
        ff_comb *p = new ff_comb(c,e,
                                 farm1->isset_cleanup_collector(),
                                 farm2->isset_cleanup_emitter());
        if (farm1->isset_cleanup_collector()) farm1->cleanup_collector(false);
        if (farm2->isset_cleanup_emitter())   farm2->cleanup_emitter(false);
        farm2->change_emitter(p, true);
    }
    farm1->remove_collector();  
    pipe.add_stage(farm1, cleanup1);
    pipe.add_stage(farm2, cleanup2);
    return pipe;
}

    
/**
 *  It combines two farms where farm1 has a default collector and 
 *  farm2 has a default emitter node. It produces a new farm whose 
 *  worker is an all-to-all building block.
 *
 */    
static inline const ff_farm combine_farms_a2a(ff_farm& farm1, ff_farm& farm2) {
    ff_farm newfarm;

    if (farm1.getCollector() != nullptr) {
        error("combine_farms, first farm has a non-default collector\n");
        return newfarm;
    }

    if (farm2.getEmitter() != nullptr) {
        error("ff_comb, second farm has a non-default emitter, use: combine_farm(farm1, emitter2, farm2)\n");
        return newfarm;
    }

    ff_a2a *a2a = new ff_a2a;
    assert(a2a);
    
    const svector<ff_node *> & w1= farm1.getWorkers();
    const svector<ff_node *> & w2= farm2.getWorkers();
    
    std::vector<ff_node*> W1(w1.size());
    std::vector<ff_node*> W2(w2.size());
    for(size_t i=0;i<W1.size();++i) W1[i]=w1[i];
    for(size_t i=0;i<W2.size();++i) W2[i]=w2[i];
    
    ff_node* emitter1 = farm1.getEmitter();
    if (emitter1) {
        newfarm.add_emitter(emitter1);
        if (farm1.isset_cleanup_emitter()) {
            newfarm.cleanup_emitter(true);
            farm1.cleanup_emitter(false);
        }
    }
    ff_node* collector2 = farm2.getCollector();
    if (farm2.hasCollector()) {
        newfarm.add_collector(collector2);
        if (farm2.isset_cleanup_collector()) {
            newfarm.cleanup_collector(true);
            farm2.cleanup_collector(false);
        }
    }        
    if (farm2.isset_cleanup_collector()) farm2.cleanup_collector(false);
    
    a2a->add_firstset(W1, farm2.ondemand_buffer(), farm1.isset_cleanup_workers());
    a2a->add_secondset(W2, farm2.isset_cleanup_workers());
    if (farm1.isset_cleanup_workers()) farm1.cleanup_workers(false);
    if (farm2.isset_cleanup_workers()) farm2.cleanup_workers(false);
    
    std::vector<ff_node*> W;
    W.push_back(a2a);
    newfarm.add_workers(W);
    if (farm1.ondemand_buffer())
        newfarm.set_scheduling_ondemand(farm1.ondemand_buffer());
    newfarm.cleanup_workers(); 
    
    return newfarm;    
}

/**
 *  It combines two farms so that the new farm produced has a single worker
 *  that is an all-to-all building block.
 *  The node passed as second parameter is composed with each worker of 
 *  the first set of workers.
 * 
 */    
template<typename E_t>     
static inline const ff_farm combine_farms_a2a(ff_farm &farm1, const E_t& node, ff_farm &farm2) {
    ff_farm newfarm;

    ff_a2a *a2a = new ff_a2a;
    assert(a2a);
	    
    const svector<ff_node *> & w1= farm1.getWorkers();
    const svector<ff_node *> & w2= farm2.getWorkers();
    
    std::vector<ff_node*> W1(w1.size());
    std::vector<ff_node*> W2(w2.size());
    for(size_t i=0;i<W1.size();++i) W1[i]=w1[i];
    for(size_t i=0;i<W2.size();++i) W2[i]=w2[i];
    
    ff_node* emitter1 = farm1.getEmitter();
    if (emitter1) newfarm.add_emitter(emitter1);
    ff_node* collector2 = farm2.getCollector();
    if (farm2.hasCollector()) newfarm.add_collector(collector2);

    std::vector<ff_node*> Wtmp(W1.size());
    for(size_t i=0;i<W1.size();++i) {
        if (!node.isMultiOutput()) {
            //const auto emitter_mo = mo_transformer(node);
            //auto c = combine_nodes(*W1[i], emitter_mo);
            //auto pc = new decltype(c)(c);
            auto mo = new internal_mo_transformer(node);
            assert(mo);
            auto pc = new ff_comb(W1[i], mo,
                                  farm1.isset_cleanup_workers() , true);    
            assert(pc);
            Wtmp[i]=pc;
        } else {
            //auto c = combine_nodes(*W1[i], node);
            //auto pc = new decltype(c)(c);
            auto e = new E_t(node);
            auto pc = new ff_comb(W1[i], e,
                                  farm1.isset_cleanup_workers(), true);    
            assert(pc);
            Wtmp[i]=pc;
        }
    }
    a2a->add_firstset(Wtmp, farm2.ondemand_buffer(), true);  // cleanup set to true
    a2a->add_secondset(W2, farm2.isset_cleanup_workers());
    if (farm2.isset_cleanup_workers()) farm2.cleanup_workers(false);    
    
    std::vector<ff_node*> W;
    W.push_back(a2a);
    newfarm.add_workers(W);
    newfarm.cleanup_workers(); // a2a will be delated at the end
    if (farm1.ondemand_buffer())
        newfarm.set_scheduling_ondemand(farm1.ondemand_buffer());
    
    return newfarm;    
}

/* 
 * This function produced the NF of two farms having the same n. of workers.
 * If the farms are ordered farm they must have the same ondemand buffer and 
 * the same ordering memory size.
 */    
static inline const ff_farm combine_farms_nf(ff_farm& farm1, ff_farm& farm2) {
    ff_farm newfarm;

    if (farm1.getNWorkers() != farm2.getNWorkers()) {
        error("combine_farms_nf, cannot combine farms with different number of workers\n");
        return newfarm;
    }
    if (farm1.isOFarm() ^ farm2.isOFarm()) {
        error("combine_farms_nf, if one of the two farms is ordered both must be ordered\n");
        return newfarm;
    }

    if (farm1.isOFarm() && farm2.isOFarm()) {
        if (farm1.ondemand_buffer() != farm2.ondemand_buffer()) {
            error("combine_farms_nf, cannot combine ordered farms with different ondemand buffer\n");
            return newfarm;
        }
        if (farm1.ordering_memory_size()!=farm2.ordering_memory_size()) {
            error("combine_farms_nf, cannot combine ordered farms with different memory size\n");
            return newfarm;
        }
    }
    const svector<ff_node *> & w1= farm1.getWorkers();
    const svector<ff_node *> & w2= farm2.getWorkers();
    
    if (w1[0]->isMultiOutput() || w2[0]->isMultiInput()) { // NOTE: we suppose homogeneous workers
        error("combine_farms_nf, cannot combine farms whose workers are either multi-output or multi-input nodes\n");
        return newfarm;
    }
    if (w1[0]->isPipe() || w2[0]->isPipe()) { // NOTE: we suppose homogeneous workers
        error("combine_farms_nf, cannot combine farms whose workers are pipeline nodes\n");
        return newfarm;
    }
    std::vector<ff_node*> W1(w1.size());
    std::vector<ff_node*> W2(w2.size());
    for(size_t i=0;i<W1.size();++i) W1[i]=w1[i];
    for(size_t i=0;i<W2.size();++i) W2[i]=w2[i];
    
    ff_node* emitter1 = farm1.getEmitter();
    if (emitter1) {
        newfarm.add_emitter(emitter1);
        if (farm1.isset_cleanup_emitter()) {
            newfarm.cleanup_emitter(true);
            farm1.cleanup_emitter(false);
        }
    }
    ff_node* collector2 = farm2.getCollector();
    if (farm2.hasCollector()) {
        newfarm.add_collector(collector2);
        if (farm2.isset_cleanup_collector()) {
            newfarm.cleanup_collector(true);
            farm2.cleanup_collector(false);
        }
    }
    
    std::vector<ff_node*> Wtmp1(W1.size());
    for(size_t i=0;i<W1.size();++i) {
        //auto c = combine_nodes(*W1[i], *W2[i]);
        //auto pc = new decltype(c)(c);
        auto pc = new ff_comb(W1[i], W2[i],
                              farm1.isset_cleanup_workers(),
                              farm2.isset_cleanup_workers());
        assert(pc);
        Wtmp1[i]=pc;
    }
    newfarm.add_workers(Wtmp1);
    newfarm.cleanup_workers();
    if (farm1.isset_cleanup_workers()) farm1.cleanup_workers(false);
    if (farm2.isset_cleanup_workers()) farm2.cleanup_workers(false);
    if (farm1.ondemand_buffer())
        newfarm.set_scheduling_ondemand(farm1.ondemand_buffer());
    if (farm1.isOFarm() || farm2.isOFarm())
        newfarm.set_ordered(farm1.ordering_memory_size());
    return newfarm;
}

/* 
 *
 * This function allows to combine two farms where the first one is an ordered farm.
 *
 *
 */
static inline const ff_pipeline combine_ofarm_farm(ff_farm& farm1, ff_farm& farm2) {
    ff_pipeline newpipe;
    if (!farm1.isOFarm()) {
        error("combine_ofarm_farm, the first farm is not an ordered farm");
        return newpipe;
    }
    // here it would be possible to call directly the combine_farms_nf function but
    // since this kind of transformation may violates the ordering semantics,
    // the user must call it explicitly
    if (farm2.isOFarm() && farm1.getNWorkers() == farm2.getNWorkers()) {   
        error("combine_ofarm_farm, two ordered farms with the same cardinality, the function cambine_farms_nf must be called explicitly\n");
        //newpipe.add_stage(combine_farms_nf(farm1,farm2));
        return newpipe;
    }
    // here we have that the first farm is an ordered farm and the second farm
    // is either a standard farm or is an ordered farm with a number of workers
    // that is different from the one of the first farm
    
    ff_farm newfarm1;

    if (farm1.ondemand_buffer())
        newfarm1.set_scheduling_ondemand(farm1.ondemand_buffer());
    
    ordered_lb* _lb= new ordered_lb(farm1.getNWorkers());
    assert(_lb);
    const size_t memsize = farm1.getNWorkers() * (2*newfarm1.ondemand_buffer()+ff_farm::DEF_IN_OUT_DIFF+3)+ DEF_OFARM_ONDEMAND_MEMORY; 
    newfarm1.ordered_resize_memory(memsize);
    _lb->init(newfarm1.ordered_get_memory(), memsize);
    newfarm1.setlb(_lb, true);
    OrderedCollectorWrapper* cw = new OrderedCollectorWrapper(DEF_OFARM_ONDEMAND_MEMORY);
    assert(cw);
    
    // emitter1 
    ff_node* emitter1 = farm1.getEmitter();   
    if (emitter1) {
        newfarm1.add_emitter(emitter1);
        if (farm1.isset_cleanup_emitter()) {
            newfarm1.cleanup_emitter(true);
            farm1.cleanup_emitter(false);
        }
    }     
    // workers1
    const svector<ff_node*>& w1= farm1.getWorkers();
    std::vector<ff_node*> W1(w1.size());
    for(size_t i=0;i<w1.size();++i) {
        W1[i] = new OrderedWorkerWrapper(w1[i], farm1.isset_cleanup_workers());
        assert(W1[i]);
    }
    if (farm1.isset_cleanup_workers())
        farm1.cleanup_workers(false);
    
    newfarm1.add_workers(W1);
    newfarm1.cleanup_workers(true);

    // collector1 + emitter2
    ff_node* collector1 = farm1.getCollector();
    ff_node* emitter2 = farm2.getEmitter();    
    if (!collector1 && !emitter2) {
        farm2.change_emitter(cw, true);
    } else {
        if (!collector1) {
            ff_comb *comb = new ff_comb(cw, emitter2, 
                                        true, farm2.isset_cleanup_emitter());
            if (farm2.isset_cleanup_emitter()) 
                farm2.cleanup_emitter(false);

            farm2.change_emitter(comb, true);
        } else {
            if (!emitter2) {
                ff_comb *comb = new ff_comb(cw, collector1, 
                                            true, farm1.isset_cleanup_collector());
                if (farm1.isset_cleanup_collector()) 
                    farm1.cleanup_collector(false);

                farm2.change_emitter(comb, true);
            } else {
                ff_comb *comb0 = new ff_comb(collector1,emitter2,
                                             farm1.isset_cleanup_collector(),
                                             farm2.isset_cleanup_emitter());
                if (farm1.isset_cleanup_collector()) 
                    farm1.cleanup_collector(false);
                if (farm2.isset_cleanup_emitter()) 
                    farm2.cleanup_emitter(false);
                
                ff_comb *comb = new ff_comb(cw, comb0,
                                            true, true);
                farm2.change_emitter(comb, true);
            }
        }
    }
    newpipe.add_stage(newfarm1);
    newpipe.add_stage(farm2);

    return newpipe;
}
    
    
/* 
 *
 * This function allows to combine two farms in several different ways
 * depending on the parameter passed to the function (node1,node2 and mergeCE):
 *
 *  case1 - node1 and node2 are both null:
 *     1. mergeCE==false:  it produces a pipeline of a single farm 
 *        whose worker is an all-to-all building block (equivalent behavior 
 *        of calling combine_farms_a2a(farm1,farm2)
 *     2. mergeCE==true:  if the parallelism degree of the two farms is 
 *        the same, it produces a pipeline of a single farm 
 *        whose workers are a composition of both farm1 and farm2 workers (normal form).
 *        If the parallelism degree of the two farms is different, we fall back 
 *        to the case1.1
 *
 *  case2 - node1 is null and node2 is not null 
 *     1. mergeCE==false: it produces a pipeline of a single farm 
 *        whose worker is an all-to-all building block where the nodes of the second set
 *        is a composition of node2 and farm2's workers
 *     2. mergeCE==true: this produces a pipeline of two farms where the first 
 *        farm has no collector while the second farm has as emitter the node2.
 *
 *  case3 - node1 is not null and node2 is null
 *     1. mergeCE==false: it produces a pipeline of a single farm 
 *        whose worker is an all-to-all building block where the nodes of the first set
 *        is a composition of farm1's workers and node1
 *     2. mergeCE==true: this produces a pipeline of two farms where the first 
 *        farm has no collector while the second farm has as emitter the node1.
 *
 *  case4 - both node1 and node2 are both not null
 *     1. mergeCE==false: it produces a pipeline of a single farm 
 *        whose worker is an all-to-all building block where the nodes of the first set
 *        is a composition of farm1's workers and node1 whereas the nodes of the second set
 *        is a composition of farm2's workers and node2.
 *     2. mergeCE==true:  this produces a pipeline of two farms where the first 
 *        farm has no collector while the second farm has as emitter the composition
 *        of node1 and node2.
 *
 * WARNING: farm1 and farm2 are passed by reference and they might be changed!
 */    
template<typename E_t, typename C_t>
static inline const ff_pipeline combine_farms(ff_farm& farm1, const C_t *node1,
                                ff_farm& farm2, const E_t *node2,
                                bool mergeCE) {
    ff_pipeline newpipe;
    
    if (mergeCE) { // we have to merge nodes!!!

        if (farm1.isOFarm() || farm2.isOFarm()) {
            if (node1!=nullptr || node2!=nullptr) {  // TODO
                error("combine_farms, FEATURE NOT YET SUPPORTED, if at least one of the two farms is an ordered farm then node1 and node2 must be nullptr\n"); 
                return newpipe;
            }
            if (farm1.getNWorkers() == farm2.getNWorkers()) {
                // here it would be possible to call directly the combine_farms_nf function but
                // since this kind of transformation may violates the ordering semantics,
                // the user must call it explicitly
                error("combine_farms, at least one of the two farms is ordered and they have the same cardinality, the function cambine_farms_nf must be called explicitly\n");
                //newpipe.add_stage(combine_farms_nf(farm1,farm2));
                return newpipe;
            }
            // the first farm is ordered 
            if (farm1.isOFarm()) {
                auto pipe = combine_ofarm_farm(farm1, farm2);
                return pipe;
            }
            // the second farm is ordered
            // here we can just remove the collector of the first farm
            farm1.remove_collector();
            newpipe.add_stage(&farm1);
            newpipe.add_stage(&farm2);
            return newpipe;
        }

        if (node2==nullptr && node1==nullptr) {  
            if (farm1.getNWorkers() == farm2.getNWorkers()) { // case1.2
                newpipe.add_stage(combine_farms_nf(farm1,farm2));
                return newpipe;
            }
            // fall back to case1.1
            // we cannot merge workers so we combine the two farms introducing
            // the all-to-all building block
            farm1.remove_collector();
            farm2.change_emitter((ff_minode*)nullptr);
            newpipe.add_stage(combine_farms_a2a(farm1,farm2));
            return newpipe;            
        }        
        if (node2!=nullptr && node1!=nullptr) {   // case4.2
            if (node1->isMultiOutput()) {
                error("combine_farms, node1 cannot be a multi-output node\n");
                return newpipe;
            }
            // here we compose node1 and node2 and we set this new 
            // node as emitter of the second farm

            farm1.remove_collector();
            auto ec= combine_nodes(*node1, *node2);
            auto pec = new decltype(ec)(ec);
            farm2.change_emitter(pec,true); // cleanup set

            newpipe.add_stage(&farm1);
            newpipe.add_stage(&farm2);
            return newpipe;
        }
        if (node1 == nullptr) {     // case2.2
            assert(node2!=nullptr);
            farm1.remove_collector();
            farm2.change_emitter((ff_minode*)nullptr);
            
            newpipe.add_stage(&farm1);
            newpipe.add_stage(&farm2);
            return newpipe;
        }
        assert(node1!=nullptr);    // case3.2
        if (node1->isMultiInput()) {  
            const struct hnode:ff_node {
                void* svc(void*in) {return in;}
            } helper_node;
            farm1.remove_collector();
            const auto comp = combine_nodes(*node1, helper_node);
            farm2.change_emitter(comp);
        } else {
            farm1.remove_collector();
            farm2.change_emitter(const_cast<C_t*>(node1));
        }
        newpipe.add_stage(&farm1);
        newpipe.add_stage(&farm2);
        return newpipe;
        
    }
    // mergeCE is false 

    if (farm1.isOFarm() || farm2.isOFarm()) {
        error("combine_farms, A2A cannot be introduced if one of the two farms is an ordered farms\n");
        return newpipe;
    }
    
    if (node2==nullptr && node1==nullptr) { // case1.1
        farm1.remove_collector();
        farm2.change_emitter((ff_minode*)nullptr);
        newpipe.add_stage(combine_farms_a2a(farm1,farm2));
        return newpipe;
    }
    if (node2!=nullptr && node1==nullptr) {
        newpipe.add_stage(combine_farms_a2a(farm1, *node2, farm2));
        return newpipe;
    }
    if (node2==nullptr && node1!=nullptr) {   // case3.1
        ff_a2a *a2a = new ff_a2a;
        if (a2a == nullptr) {
            error("combine_farms, FATAL ERROR, not enough memory\n");
            return newpipe;
        }        
        ff_farm newfarm;
        const svector<ff_node *> & w1= farm1.getWorkers();
        const svector<ff_node *> & w2= farm2.getWorkers();
        
        std::vector<ff_node*> W1(w1.size());
        std::vector<ff_node*> W2(w2.size());
        for(size_t i=0;i<W1.size();++i) W1[i]=w1[i];
        for(size_t i=0;i<W2.size();++i) W2[i]=w2[i];
        
        ff_node* emitter1 = farm1.getEmitter();
        if (emitter1) {
            newfarm.add_emitter(emitter1);
            if (farm1.isset_cleanup_emitter()) {
                newfarm.cleanup_emitter(true);
                farm1.cleanup_emitter(false);
            }
        }
        ff_node* collector2 = farm2.getCollector();
        if (farm2.hasCollector()) {
            newfarm.add_collector(collector2);
            if (farm2.isset_cleanup_collector()) {
                newfarm.cleanup_collector(true);
                farm2.cleanup_collector(false);
            }
        }

        std::vector<ff_node*> Wtmp1(W1.size());
        for(size_t i=0;i<W1.size();++i) {
            if (!node1->isMultiOutput()) {
                //const auto collector_mo = mo_transformer(*node1);
                //auto c = combine_nodes(*W1[i], collector_mo);
                //auto pc = new decltype(c)(c);
                auto mo = new internal_mo_transformer(*node1);
                assert(mo);
                auto pc = new ff_comb(W1[i], mo,
                                      farm1.isset_cleanup_workers() , true);
                assert(pc);
                Wtmp1[i]=pc;
            } else {
                //auto c = combine_nodes(*W1[i], *node1);
                //auto pc = new decltype(c)(c);
                auto c = new C_t(*node1);
                assert(c);
                auto pc = new ff_comb(W1[i], c,
                                      farm1.isset_cleanup_workers(), true);
                assert(pc);
                Wtmp1[i]=pc;
            }
        }
        if (farm1.isset_cleanup_workers()) farm1.cleanup_workers(false);
        a2a->add_firstset(Wtmp1, farm2.ondemand_buffer(), true);  // cleanup set to true
        a2a->add_secondset(W2, farm2.isset_cleanup_workers());
        if (farm2.isset_cleanup_workers()) farm2.cleanup_workers(false);
    
        std::vector<ff_node*> W;
        W.push_back(a2a);
        newfarm.add_workers(W);
        newfarm.cleanup_workers(); // a2a will be delated at the end
        if (farm1.ondemand_buffer())
            newfarm.set_scheduling_ondemand(farm1.ondemand_buffer());
        
        newpipe.add_stage(newfarm);
        return newpipe;
    }
    assert(node2!=nullptr && node1!=nullptr);    // case4.1

    // TODO: we can relax the following two constraints.
    if (node1->isMultiInput()) {
        error("combine_farms, node1 cannot be a multi-input node\n");
        return newpipe;
    }
    if (node2->isMultiOutput()) {
        error("combine_farms, node2 cannot be a multi-output node\n");
        return newpipe;
    }

    ff_a2a *a2a = new ff_a2a;
    if (a2a == nullptr) {
        error("combine_farms, FATAL ERROR, not enough memory\n");
        return newpipe;
    }

    ff_farm newfarm;
    const svector<ff_node *> & w1= farm1.getWorkers();
    const svector<ff_node *> & w2= farm2.getWorkers();
    
    std::vector<ff_node*> W1(w1.size());
    std::vector<ff_node*> W2(w2.size());
    for(size_t i=0;i<W1.size();++i) W1[i]=w1[i];
    for(size_t i=0;i<W2.size();++i) W2[i]=w2[i];
    
    ff_node* emitter1 = farm1.getEmitter();
    if (emitter1) {
        newfarm.add_emitter(emitter1);
        if (farm1.isset_cleanup_emitter()) {
            newfarm.cleanup_emitter(true);
            farm1.cleanup_emitter(false);
        }        
    }
    ff_node* collector2 = farm2.getCollector();
    if (farm2.hasCollector()) {
        newfarm.add_collector(collector2);
        if (farm2.isset_cleanup_collector()) {
            newfarm.cleanup_collector(true);
            farm2.cleanup_collector(false);
        }
    }

    std::vector<ff_node*> Wtmp1(W1.size());
    for(size_t i=0;i<W1.size();++i) {
        if (!node1->isMultiOutput()) {
            //const auto collector_mo = mo_transformer(*node1);
            //auto c = combine_nodes(*W1[i], collector_mo);
            //auto pc = new decltype(c)(c);
            auto mo = new internal_mo_transformer(*node1);
            assert(mo);
            auto pc = new ff_comb(W1[i], mo, 
                                  farm1.isset_cleanup_workers() , true);
            Wtmp1[i]=pc;
        } else {
            //auto c = combine_nodes(*W1[i], *node1);
            //auto pc = new decltype(c)(c);
            auto c = new C_t(*node1);
            assert(c);
            auto pc = new ff_comb(W1[i], c,
                                  farm1.isset_cleanup_workers(), true);
            Wtmp1[i]=pc;
        }
    }
    if (farm1.isset_cleanup_workers()) farm1.cleanup_workers(false);
    a2a->add_firstset(Wtmp1, farm2.ondemand_buffer(), true);  // cleanup set to true

    std::vector<ff_node*> Wtmp2(W2.size());
    for(size_t i=0;i<W2.size();++i) {
        if (!node2->isMultiInput()) {
            //const auto emitter_mi = mi_transformer(*node2);
            //auto c = combine_nodes(emitter_mi, *W2[i]);
            //auto pc = new decltype(c)(c);
            auto mi = new internal_mi_transformer(*node2);
            assert(mi);
            auto pc = new ff_comb(mi,W2[i],
                                  true, farm2.isset_cleanup_workers());
            assert(pc);
            Wtmp2[i]=pc;
        } else {
            //auto c = combine_nodes(*node2, *W2[i]);
            //auto pc = new decltype(c)(c);
            auto e = new E_t(*node2);
            assert(e);
            auto pc = new ff_comb(e, W2[i],
                                  true, farm2.isset_cleanup_workers());
            assert(pc);
            Wtmp2[i]=pc;
        }
    }
    if (farm2.isset_cleanup_workers()) farm2.cleanup_workers(false);
    a2a->add_secondset(Wtmp2, true);  // cleanup set to true
    
    std::vector<ff_node*> W;
    W.push_back(a2a);
    newfarm.add_workers(W);
    newfarm.cleanup_workers(); // a2a will be deleted at the end
    if (farm1.ondemand_buffer())
        newfarm.set_scheduling_ondemand(farm1.ondemand_buffer());
    
    newpipe.add_stage(newfarm);
    return newpipe;;
}


    
} // namespace ff 
#endif /* FF_COMBINE_HPP */
