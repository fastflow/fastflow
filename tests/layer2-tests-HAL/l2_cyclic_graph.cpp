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
 *       +-----unbound-----+
 *       v                 +
 *      [e1]---->[n1]---->[n5]
 *       +
 *       +------>[n2]------+ 
 *                         v     
 *                        [c1]              
 *                         ^
 *       +------>[n3]------+
 *       +
 *      [e2]---->[n4]---->[n6]
 *       ^                 +
 *       +-----unbound-----+ 
 *
 */

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
    N n1(1),n2(2),n3(3),n4(4),n5(5),n6(6);
    n1.create_input_buffer(100);
    n2.create_input_buffer(100);
    n3.create_input_buffer(100);
    n4.create_input_buffer(100);
    n5.create_input_buffer(100);
    n6.create_input_buffer(100);

    // Create 2 emitters e1,e2 and the links 
    // e1->n1, e1->n2, e2->n3, e2->n4
    svector<ff_node*> we1;
    we1.push_back(&n1);
    we1.push_back(&n2);
    svector<ff_node*> we2;
    we2.push_back(&n3);
    we2.push_back(&n4);
    E e1(we1,1 /* id */,ntasks,niterations), e2(we2,2 /* id */,ntasks,niterations);
    e1.skipfirstpop(true);
    e2.skipfirstpop(true);
    // Create an inut channel for e1 and e2. They unbound toa void deadlocks
    e1.create_input_buffer(10,false); 
    e2.create_input_buffer(10,false); 

    // link n1->n5 and n4->n6
    n1.set_output_buffer(n5.get_in_buffer());
    n4.set_output_buffer(n6.get_in_buffer());
    
    // These are feedback channels
    n6.set_output_buffer(e2.get_in_buffer());
    n5.set_output_buffer(e1.get_in_buffer());

    // Create 1 collector and the links
    // n2->c1, n3->c1
    
    n2.create_output_buffer(100);
    n3.create_output_buffer(100);
    svector<ff_node*> wc1;
    wc1.push_back(&n2);
    wc1.push_back(&n3);
    C c1(wc1,1 /* id */);
    printf("c1 return %d\n",c1.create_input_buffer(10)); 
    printf("Init done\n");

    n1.run();
    n2.run();
    n3.run();
    n4.run();
    n5.run();
    n6.run();
    e1.run();
    e2.run();
    c1.run();
    printf("All running\n");

    e1.wait();
    e2.wait();
    n1.wait();
    n2.wait();
    n3.wait();
    n4.wait();
    n5.wait();
    n6.wait();
    c1.wait();

    return 0;
}
