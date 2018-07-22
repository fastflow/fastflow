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
#include <ff/farm.hpp>

namespace ff {

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
    template<typename T1, typename T2>
    ff_comb(T1* n1, T2* n2, bool first_cleanup=false, bool second_cleanup=false){
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

    // NOTE: it is multi-input only if the first node is multi-input
    inline bool isMultiInput() const {
        if (comp_nodes[0]->isMultiInput()) return true;
        return comp_multi_input;
    }
    inline bool isMultiOutput() const {
        if (comp_nodes[1]->isMultiOutput()) return true;
        return comp_multi_output;
    }

    inline bool isComp() const        { return true; }
    
protected:
    ff_comb():ff_minode() {}

    template<typename T1, typename T2>
    inline bool check(T1* n1, T2* n2) {
        if (n1->isFarm() || n1->isAll2All() || n1->isPipe() ||
            n2->isFarm() || n2->isAll2All() || n2->isPipe()) {
            error("COMBINE, input nodes cannot be farm, all-2-all or pipeline building-blocks\n");
            return false;
        }        
        if (n1->isMultiInput() && n2->isMultiInput() ) {
            error("COMBINE, both nodes cannot be multi-input nodes\n");
            return false;
        }
        if (n1->isMultiOutput() && n2->isMultiOutput() ) {
            error("COMP, both nodes cannot be multi-output nodes\n");
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
            ((ff_minode*)n1)->setgt(ff_minode::getgt());

            if (ff_minode::prepare()<0) return -1;
        }

        // registering a special callback if the last stage does
        // not have an output channel
        ff_node *n2 = getLast();
        if (!n2->isMultiOutput() && (n2->get_out_buffer() == nullptr))
            n2->registerCallback(devnull, nullptr);   // devnull callback

        
        prepared = true;
        return 0;
    }

    void set_multiinput() {
        // see farm.hpp
        // to avoid that in the eosnotify the EOS is propogated
        if (comp_nodes[0]->isComp())
            return comp_nodes[0]->set_multiinput();
        comp_multi_input=true;
    }
    void set_multioutput() {
        // see farm.hpp
        // to avoid that in the eosnotify the EOS is propagated
        // NOTE that propagateEOS for a standard node is nop.
        if (comp_nodes[1]->isComp())
            return comp_nodes[1]->set_multioutput();
        comp_multi_output=true;
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
                        comp_nodes[1]->propagateEOS();         
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
                        comp_nodes[1]->propagateEOS();
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
                r1= comp_nodes[0]->svc(task);
                if (!(r1 == FF_GO_ON || r1 == FF_GO_OUT || r1 == FF_EOS_NOFREEZE))
                    comp_nodes[1]->push_comp_local(r1);
                if (r1 == FF_EOS) ret=FF_GO_OUT;
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
        if (comp_nodes[0]->isComp()) return comp_nodes[0]->set_input(w);
        //assert(comp_nodes[0]->isMultiInput());
        if (comp_nodes[0]->set_input(w)<0) return -1;
        // if the first node of the comp is a multi-input node
        // we have to set the input of the current ff_minode that
        // is implementing the composition
        return ff_minode::set_input(w);        
    }
    int set_input(ff_node *n) {
        if (comp_nodes[0]->isComp()) return comp_nodes[0]->set_input(n);
        //assert(comp_nodes[0]->isMultiInput());
        if (comp_nodes[0]->set_input(n)<0) return -1;
        // if the first node of the comp is a multi-input node
        // we have to set the input of the current ff_minode that
        // is implementing the composition
        return ff_minode::set_input(n);        
    }
    int set_input_feedback(ff_node *n) {
        if (comp_nodes[0]->isComp()) return comp_nodes[0]->set_input_feedback(n);
        assert(comp_nodes[0]->isMultiInput());
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
   
    int run(bool=false) {
        if (!prepared) if (prepare()<0) return -1;

        // set blocking mode for the last node of the composition
        getLast()->blocking_mode(blocking_in);      
        if (comp_nodes[0]->isMultiInput())
            return ff_minode::run();
        
        if (ff_node::run(true)<0) return -1;
        return 0;
    }

     int  wait(/* timeout */) {
         if (comp_nodes[0]->isMultiInput())
             return ff_minode::wait();
         if (ff_node::wait()<0) return -1;
         return 0;
    }

    void eosnotify(ssize_t id=-1) {
        ++neos;
        comp_nodes[0]->eosnotify(id);

        // the eosnotify might produce some data in output so we have to call the svc
        // of the next stage
        void *ret = svc_comp_node1(nullptr, GO_ON);

        // NOTE: if the first node is multi-input the EOS is not propagated        
        if (comp_nodes[0]->isMultiInput() ||
            comp_multi_input) {

            if (comp_nodes[0]->isMultiInput()) {
                ff_minode* mi = reinterpret_cast<ff_minode*>(comp_nodes[0]);
                if (neos >= mi->get_num_inchannels() &&
                    (ret != FF_GO_OUT) /*this means that svc_comp_node1 has already called eosnotify */
                    )
                    comp_nodes[1]->eosnotify(id);                
            } 
            return;
        }
        if (ret != FF_GO_OUT)
            comp_nodes[1]->eosnotify(id);
        
        if (comp_nodes[1]->isComp()) return;
        
        if (comp_nodes[1]->isMultiOutput() ||
            comp_multi_output)
            comp_nodes[1]->propagateEOS();
        else
            comp_nodes[1]->ff_send_out(FF_EOS);
    }

    void propagateEOS() {
        if (comp_nodes[1]->isComp()) {
            comp_nodes[1]->propagateEOS();
            return;
        }
        
        if (comp_nodes[1]->isMultiOutput() ||
            comp_multi_output)               
            comp_nodes[1]->propagateEOS();
        else
            comp_nodes[1]->ff_send_out(FF_EOS);
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

    int create_input_buffer(int nentries, bool fixedsize=true) {
        if (isMultiInput())
            return ff_minode::create_input_buffer(nentries,fixedsize);
        return ff_node::create_input_buffer(nentries,fixedsize);
    }
    int create_output_buffer(int nentries, bool fixedsize=true) {
        return comp_nodes[1]->create_output_buffer(nentries,fixedsize);
    }
    FFBUFFER * get_in_buffer() const {
        if (getFirst()->isMultiInput()) return nullptr;
        return ff_node::get_in_buffer();
    }

    int set_output_buffer(FFBUFFER * const o) {
        return comp_nodes[1]->set_output_buffer(o);
    }

    // a composition can be passed as filter to a farm emitter  
    void setlb(ff_loadbalancer *elb) {
        comp_nodes[1]->setlb(elb);
    }
    // a composition can be passed as filter to a farm collector
    void setgt(ff_gatherer *egt) {
        comp_nodes[0]->setgt(egt);
        ff_minode::setgt(egt);
    }

    // consumer
    bool init_input_blocking(pthread_mutex_t   *&m,
                             pthread_cond_t    *&c,
                             std::atomic_ulong *&counter) {        
        ff_node *n = getFirst();
        if (n->isMultiInput()) {
            return ff_minode::init_input_blocking(m,c,counter);
        }
        return ff_node::init_input_blocking(m,c,counter);
    }
    void set_input_blocking(pthread_mutex_t   *&m,
                            pthread_cond_t    *&c,
                            std::atomic_ulong *&counter) {
        if (this->isMultiInput()) {
            ff_minode::set_input_blocking(m,c,counter);
            return;
        }
        return ff_node::set_input_blocking(m,c,counter);        
    }    

    // producer
    bool init_output_blocking(pthread_mutex_t   *&m,
                              pthread_cond_t    *&c,
                              std::atomic_ulong *&counter) {
        return comp_nodes[1]->init_output_blocking(m,c,counter);
    }
    void set_output_blocking(pthread_mutex_t   *&m,
                             pthread_cond_t    *&c,
                             std::atomic_ulong *&counter) {
        comp_nodes[1]->set_output_blocking(m,c,counter);
    }

    // the following calls are needed because a composition
    // uses as output channel(s) the one(s) of the second node.
    // these functions should not be called if the node is multi-output
    inline bool  get(void **ptr)                 { return comp_nodes[1]->get(ptr);}
    inline pthread_mutex_t   &get_prod_m()       { return comp_nodes[1]->get_prod_m();}
    inline pthread_cond_t    &get_prod_c()       { return comp_nodes[1]->get_prod_c();}
    inline std::atomic_ulong &get_prod_counter() { return comp_nodes[1]->get_prod_counter();}
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
    size_t neos=0;
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
const ff_comb combine_nodes(T1& n1, T2& n2) {
    ff_comb comp;
    comp.add_node(n1,n2);
    return comp;
}
    
/**
 *  combines either basic nodes or ff_comb(s) and returns a unique_ptr
 *  useful to add ff_comb as farm's workers
 */    
template<typename T1, typename T2>    
std::unique_ptr<ff_node> unique_combine_nodes(T1& n1, T2& n2) {
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
const ff_pipeline combine_nodes_in_pipeline(ff_node& node1, ff_node& node2, bool cleanup1=false, bool cleanup2=false) {
    if (node1.isAll2All() || node2.isAll2All()) {
        error("combine_nodes_in_pipeline, cannot be used if one of the nodes is A2A\n");
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
const ff_farm combine_farms_a2a(ff_farm& farm1, ff_farm& farm2) {
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
const ff_farm combine_farms_a2a(ff_farm &farm1, const E_t& node, ff_farm &farm2) {
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
            auto pc = new ff_comb(W1[i], (ff_node*)(new mo_transformer(node)),
                                  farm1.isset_cleanup_workers() , true);    
            assert(pc);
            Wtmp[i]=pc;
        } else {
            //auto c = combine_nodes(*W1[i], node);
            //auto pc = new decltype(c)(c);
            auto pc = new ff_comb(W1[i], (ff_node*)(new E_t(node)),
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
    newfarm.set_scheduling_ondemand(farm1.ondemand_buffer());
    
    return newfarm;    
}

/* 
 * This function produced the NF of two farms having the same n. of workers.
 */    
const ff_farm combine_farms(ff_farm& farm1, ff_farm& farm2) {
    ff_farm newfarm;

    if (farm1.getNWorkers() != farm2.getNWorkers()) {
        error("combine_farms, cannot combine farms with different number of workers\n");
        return newfarm;
    }
    const svector<ff_node *> & w1= farm1.getWorkers();
    const svector<ff_node *> & w2= farm2.getWorkers();
    
    if (w1[0]->isMultiOutput() || w2[0]->isMultiInput()) {
        error("combine_farms, cannot combine farms whose workers are either multi-output or multi-input nodes\n");
        return newfarm;
    }
    std::vector<ff_node*> W1(w1.size());
    std::vector<ff_node*> W2(w2.size());
    for(size_t i=0;i<W1.size();++i) W1[i]=w1[i];
    for(size_t i=0;i<W2.size();++i) W2[i]=w2[i];
    
    ff_node* emitter1 = farm1.getEmitter();
    if (emitter1) newfarm.add_emitter(emitter1);
    ff_node* collector2 = farm2.getCollector();
    if (farm2.hasCollector()) newfarm.add_collector(collector2);
    
    std::vector<ff_node*> Wtmp1(W1.size());
    for(size_t i=0;i<W1.size();++i) {
        auto c = combine_nodes(*W1[i], *W2[i]);
        auto pc = new decltype(c)(c);
        Wtmp1[i]=pc;
    }
    newfarm.add_workers(Wtmp1);
    newfarm.cleanup_workers();
    newfarm.set_scheduling_ondemand(farm1.ondemand_buffer());
    
    return newfarm;
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
 *        whose workers are a composition of both farm1 and farm2 workers.
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
const ff_pipeline combine_farms(ff_farm& farm1, const C_t *node1,
                                ff_farm& farm2, const E_t *node2,
                                bool mergeCE) {
    ff_pipeline newpipe;
    
    if (mergeCE) { // we have to merge nodes!!!
        if (node2==nullptr && node1==nullptr) {  
            if (farm1.getNWorkers() == farm2.getNWorkers()) { // case1.2
                newpipe.add_stage(combine_farms(farm1,farm2));
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
        
    }  // mergeCE is false  -------------------------------------------------
    
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
                auto pc = new ff_comb(W1[i], (ff_node*)(new mo_transformer(*node1)),
                                      farm1.isset_cleanup_workers() , true);
                assert(pc);
                Wtmp1[i]=pc;
            } else {
                //auto c = combine_nodes(*W1[i], *node1);
                //auto pc = new decltype(c)(c);
                auto pc = new ff_comb(W1[i], (ff_node*)(new C_t(*node1)),
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
        newfarm.set_scheduling_ondemand(farm1.ondemand_buffer());
        
        newpipe.add_stage(newfarm);
        return newpipe;
    }
    assert(node2!=nullptr && node1!=nullptr);    // case4.1
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
            auto pc = new ff_comb(W1[i], (ff_node*)(new mo_transformer(*node1)),
                                  farm1.isset_cleanup_workers() , true);
            Wtmp1[i]=pc;
        } else {
            //auto c = combine_nodes(*W1[i], *node1);
            //auto pc = new decltype(c)(c);
            auto pc = new ff_comb(W1[i], (ff_node*)(new C_t(*node1)),
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
            auto pc = new ff_comb((ff_node*)(new mi_transformer(*node2)),W2[i],
                                  true, farm2.isset_cleanup_workers());
            
            Wtmp2[i]=pc;
        } else {
            //auto c = combine_nodes(*node2, *W2[i]);
            //auto pc = new decltype(c)(c);
            auto pc = new ff_comb((ff_node*)(new E_t(*node2)), W2[i],
                                  true, farm2.isset_cleanup_workers());            
            Wtmp2[i]=pc;
        }
    }
    if (farm2.isset_cleanup_workers()) farm2.cleanup_workers(false);
    a2a->add_secondset(Wtmp2, true);  // cleanup set to true
    
    std::vector<ff_node*> W;
    W.push_back(a2a);
    newfarm.add_workers(W);
    newfarm.cleanup_workers(); // a2a will be deleted at the end
    newfarm.set_scheduling_ondemand(farm1.ondemand_buffer());
    
    newpipe.add_stage(newfarm);
    return newpipe;;
}


    
} // namespace ff 
#endif /* FF_COMBINE_HPP */
