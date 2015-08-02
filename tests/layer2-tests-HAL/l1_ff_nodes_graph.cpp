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
/*
 *  Layer 2 test: graph of nodes.
 *
 *
 *    -----            ------            ------
 *   |  A  |--------> |  B   |--------> |  D   |
 *   |     |          |      |          |      |
 *    -----            ------            ------
 *        \              |
 *         \             |
 *          \            | 
 *           \           v
 *            \       ------
 *             \-->  |   C  |
 *                   |      |
 *                    ------ 
 */
#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/spin-lock.hpp>
#include <ff/svector.hpp>

using namespace ff;

class A: public ff_monode {
public:

    A(svector<ff_node*>& w,long ntasks):ntasks(ntasks) {
        set_output(w);
    }

    int svc_init() {
        printf("A initialised: sending %ld tasks\n",ntasks);
        return 0;
    }

    void* svc(void *) {
        for(long i=1;i<=ntasks;++i) 
            ff_send_out((void*)i);

        return NULL;
    }

    int wait() { 
        return ff_monode::wait();
    }

private:
    long ntasks;
};


class B: public ff_monode {
public:
    B(svector<ff_node*>& w) { 
        set_output(w);
    }

    int svc_init() {
        printf("B initialised: should receive half of the tasks (+/-1)\n");
        return 0;
    }

    void* svc(void *t) {
        long data = (long) t;
        printf("B received one task: %ld\n",data);
        return t;                
    }

    int wait() { return ff_monode::wait(); }

    int create_input_buffer(int nentries, bool fixedsize=true) {
        return ff_node::create_input_buffer(nentries, fixedsize);
    }
};

class C: public ff_node {
public:
    int svc_init() {
        init_unlocked(lock);
        received = 0;
        printf("C initialised: it will terminate at the first null");
        return 0;
    }
    
    void svc_end() {
        printf("C received %ld tasks\n",received);
    }

    void* svc(void *t) {
        long data = (long) t;
        printf("C received one task: %ld\n",data);
        ++received;
        return GO_ON;
    }
    
    inline bool  put(void * ptr) {
        spin_lock(lock);
        bool r = ff_node::put(ptr);
        spin_unlock(lock);
        return r;
        return false;
    }
    
    virtual int run(bool=false)   { return ff_node::run(); }
    int wait()  { return ff_node::wait(); }
    int create_input_buffer(int nentries, bool fixedsize=true) {
        return ff_node::create_input_buffer(nentries, fixedsize);
    }

protected:
    lock_t lock;
private:
    long received;
};

class D: public ff_node {
public:
    void* svc(void *t) {
        long data = (long) t;
        printf("D received one task: %ld\n",data);
        return GO_ON;
    }

    virtual int run(bool=false)  { return ff_node::run(); }
    int wait() { return ff_node::wait(); }
    int create_input_buffer(int nentries, bool fixedsize=true) {
        return ff_node::create_input_buffer(nentries, fixedsize);
    }
};

int main() {
    long ntasks = 20;
    C c;
    c.create_input_buffer(100);

    D d;
    d.create_input_buffer(100);

    svector<ff_node*> wB;
    wB.push_back(&d);
    wB.push_back(&c);
    B b(wB);
    b.create_input_buffer(100);
    svector<ff_node*> wA;
    wA.push_back(&b);
    wA.push_back(&c);
    A a(wA,ntasks);
    a.skipfirstpop(true);

    d.run();
    c.run();
    b.run();
    a.run();

    a.wait();
    b.wait();
    c.wait();
    d.wait();

    return 0;
}
