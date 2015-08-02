/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
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
/*  Author: aldinuc
 *  Layer 2 test: graph of nodes, 
 *                e1,e2 = multiple output nodes (ff_monode)
 *                n1,n2,n3,n4,n5,6 = nodes (ff_node)
 *                c1 = multiple input node (ff_minode)
 *
 *       +----------unbound-----+
 *       v                      +
 *      (e[0])---->(n[0])---->(n[4])
 *       +
 *       +-------->(n[1])-------+ 
 *                              v     
 *                            (c[0])              
 *                              ^
 *       +-------->(n[2])-------+
 *       +
 *      (e[1])---->(n[3])---->(n[5])
 *       ^                      +
 *       +----------unbound-----+ 
 *
 *  This is almost the same code of l2_cyclic_graph with monitoring enabled
 *  and a sligtly different coding style. The usage of vectors for e[], n[], 
 *  c[] demonstrates that the building of the graph can be easily automated
 *  and parametrised.
 */

#define TRACE_FASTFLOW 1 // This enables run-time monitor

#include <vector>
#include <iostream>
#include <ff/node.hpp>
#include <ff/farm.hpp>

using namespace ff;


class E: public ff_monode {
public:

    E(svector<ff_node*>& w,long id, long ntasks, long niterations):id(id),ntasks(ntasks),niterations(niterations) {
        set_output(w);
    }
    
    int svc_init() {
        printf("E1 initialised: sending %ld tasks\n",ntasks);
        return 0;
    }

     void svc_end() {
        printf("E%ld terminating\n",id);
    }


    void* svc(void *) {
        
        for(long i=1;i<=ntasks;++i) 
            ff_send_out((void*)i);
        if(--niterations<=0) return NULL;
        else return GO_ON;                
    }

    int create_input_buffer(int nentries, bool fixedsize=true) {
        return ff_monode::create_input_buffer(nentries, fixedsize);
    }

    int wait() { 
        return ff_monode::wait(); 
    }

private:
    long id;
    long ntasks;
    long niterations;
};


class N: public ff_node {
public:

    N(long id):id(id) {}

    int svc_init() {
        received = 0;
        printf("N%ld initialised: it will terminate at the first NULL\n",id);
        return 0;
    }
    
    void svc_end() {
        printf("N%ld terminating, got %ld tasks\n",id,received);
    }

    void* svc(void *t) {
        long data = (long) t;
        printf("N%ld received task [%ld]\n",id,data);
        ++received;
        return t;
    }
 
    virtual int run(bool=false)   { return ff_node::run(); }
    int wait()  { return ff_node::wait(); }
    int create_input_buffer(int nentries, bool fixedsize=true) {
        return ff_node::create_input_buffer(nentries, fixedsize);
    }
    int create_output_buffer(int nentries, bool fixedsize=true) {
        return ff_node::create_output_buffer(nentries, fixedsize);
    }
    virtual int set_output_buffer(FFBUFFER * const o) {
        return ff_node::set_output_buffer(o); 
    }

private:
    long id;
    long received;
};



class C: public ff_minode {
public:

    C(svector<ff_node*>& w,long id):id(id) {
        set_input(w);
    }
    
    int svc_init() {
        printf("C%ld initialised: receiving tasks\n",id);
        received=0;
        return 0;
    }

    void svc_end() {
        printf("C%ld terminating, got %ld tasks\n",id,received);
    }

    void* svc(void *t) {
        long data = (long) t;
        printf("C%ld received a task [%ld]\n",id,data);
        ++received;
        return GO_ON;                
    }

    int create_input_buffer(int nentries, bool fixedsize=true) {
        return ff_minode::create_input_buffer(nentries, fixedsize);
    }

    int wait() { 
        return ff_minode::wait(); 
    }
   
private:
    long id;
    long received;
};


int main(int argc, char ** argv) {
    
    if (argc < 3) {
        printf("usage: l2_cyclic_graph ntasks niterations\n");
        return 0;
    }
    
    long ntasks = atol(argv[1]);
    long niterations = atol(argv[2]);
    
    
    // Crete 6 generic nodes - 1 in channel 1 out channel
    svector<N*> n;
    for (int i=0;i<6;++i) {
        N *p = new N(i);
        p->create_input_buffer(100);
        n.push_back(p);
    }
    // Create 2 emitters e1,e2 and the links 
    // e1->n1, e1->n2, e2->n3, e2->n4
    svector<ff_node*> we[2];
    we[0].push_back(n[0]);
    we[0].push_back(n[1]);
    we[1].push_back(n[2]);
    we[1].push_back(n[3]);
    svector<E*> e;
    for (int i=0;i<2;++i) {
        E *p = new E(we[i],i /* id */,ntasks,niterations);
        // Emitters starts the game, they do not wait for the first input
        p->skipfirstpop(true);
        // Create an inut channel for e[1]. They unbound toa void deadlocks
        p->create_input_buffer(10,false);
        e.push_back(p);
    }

    // link n1->n5 and n4->n6
    n[0]->set_output_buffer(n[4]->get_in_buffer());
    n[3]->set_output_buffer(n[5]->get_in_buffer());
    
    // These are feedback channels
    n[5]->set_output_buffer(e[0]->get_in_buffer());
    n[4]->set_output_buffer(e[1]->get_in_buffer());

    // Create 1 collector and the links
    // n2->c1, n3->c1
    
    n[1]->create_output_buffer(100);
    n[2]->create_output_buffer(100);
    svector<ff_node*> wc;
    wc.push_back(n[1]);
    wc.push_back(n[2]);
    svector<C*> c;
    c.push_back(new C(wc,0 /* id */));
    c[0]->create_input_buffer(10);
 
    printf("Init done\n");

    for (int i=0;i<6;++i) {
        n[i]->run();
    }
    for (int i=0;i<2;++i)
        e[i]->run();
    c[0]->run();

    printf("All running\n");

    // Waiting for all to complete. All nodes should be run before waiting.
    for (int i=0;i<2;++i)
        e[i]->wait();
    for (int i=0;i<6;++i)
        n[i]->wait();
    c[0]->wait();

    // Print stats
    for (int i=0;i<6;++i) {
        printf("Node N%d\n",i);
        n[i]->ffStats(std::cout);
    }
    for (int i=0;i<2;++i) {
        printf("Emitter E%d\n",i);
        e[i]->ffStats(std::cout);
    }
    printf("Collector C%d\n",0);
    c[0]->ffStats(std::cout);
    return 0;
}
