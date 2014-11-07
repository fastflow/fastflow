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

/* Author: Massimo Torquati
 * 
 * Just a funny test using the low-level features of the FastFlow framework.
 *
 *    -----            ------            ------             ------ 
 *   |  A  |--------> |  B   |--------> |  C   |---------> |  D   | 
 *   |     |          |      |          |      |           |      |
 *    -----            ------            ------             ------
 *        ^              |                   ^                |
 *         \             |                    \               |
 *          -------------                      ---------------
 *
 */

#include <ff/node.hpp>
#include <ff/farm.hpp>  // for ff_minode ff_monode
#include <ff/svector.hpp>

using namespace ff;

#define NUMTASKS 30

/* --------------------------------------- */
class A: public ff_node {
public:
    void *svc(void *task) {
        static long k=1;
        if (task==NULL) {
            ff_send_out((void*)k);
            return GO_ON;
        }
        printf("A received %ld back from B\n", (long)task);
        if (k++==NUMTASKS) return NULL;
        return (void*)k;
    }
    // the following ones are protected in ff_node
    int create_input_buffer(int nentries, bool fixedsize=true) {
        return ff_node::create_input_buffer(nentries,fixedsize);
    }
    int set_output_buffer(FFBUFFER * const o) {
        return ff_node::set_output_buffer(o);
    }
    int run(int myid)   { 
        ff_node::set_id(myid);
        return ff_node::run(); 
    }
    int wait()  { return ff_node::wait(); }
    void skipfirstpop(bool sk)   { ff_node::skipfirstpop(sk);}
};
/* --------------------------------------- */
class B: public ff_monode {
public:
    void *svc(void *task) {
        const long t = (long)task;
        printf("B got %ld from A\n", t);
        if (t & 0x1) ff_send_out_to(task, 1); 
        ff_send_out_to(task, 0); 
        return GO_ON;
    }

    int create_input_buffer(int nentries, bool fixedsize=true) {
        return ff_node::create_input_buffer(nentries,fixedsize);
    }    
    int set_output_buffer(FFBUFFER * const o) {
        return ff_node::set_output_buffer(o);
    }
    int run(int myid)   { 
        ff_node::set_id(myid);
        return ff_monode::run(); 
    }
    int wait()  { return ff_monode::wait(); }
};
/* --------------------------------------- */
class C: public ff_minode {
public:
    void *svc(void *task) {
        if (get_channel_id() < 0 ) {
            printf("C sending task %ld to D\n", (long)task);
            return task;
        }
        printf("C got task=%ld from D (channelid=%ld)\n", (long)task, get_channel_id());
        return GO_ON;
    }
    // this method is called when the EOS is received from the input channels
    void eosnotify(ssize_t channel_id) {
        if (channel_id < 0) {
            printf("C received EOS from %ld, sending EOS to D\n", channel_id);
            ff_send_out((void*)FF_EOS);
        }
    }
    
    int create_input_buffer(int nentries, bool fixedsize=true) {
        return ff_minode::create_input_buffer(nentries,fixedsize);
    }
    int set_output_buffer(FFBUFFER * const o) {
        return ff_node::set_output_buffer(o);
    }
    int run(int myid)   { 
        ff_node::set_id(myid);
        return ff_minode::run(); 
    }
    int wait()  { return ff_minode::wait(); }
};
/* --------------------------------------- */

class D: public ff_node {
public:
    void *svc(void *t) { 
        printf("D got %ld, sending it back\n", (long)t);
        fflush(stdout);
        return t;
    }
    int create_input_buffer(int nentries, bool fixedsize=true) {
        return ff_node::create_input_buffer(nentries,fixedsize);
    }
    int set_output_buffer(FFBUFFER * const o) {
        return ff_node::set_output_buffer(o);
    }
    int run(int myid)   { 
        ff_node::set_id(myid);
        return ff_node::run(); 
    }
    int wait()  { return ff_node::wait(); }
};
/* --------------------------------------- */

int main() {
    A a;  B b;
    C c;  D d;
    ff_buffernode buf(100); // this is just a buffer

    // create input buffer for each node
    a.create_input_buffer(100);
    b.create_input_buffer(100);
    c.create_input_buffer(100);
    d.create_input_buffer(100);
    
    //set output buffer for each node
    a.set_output_buffer(b.get_in_buffer());
    svector<ff_node*> w;
    w.push_back(&a);
    w.push_back(&buf);
    b.set_output(w);
    c.set_output_buffer(d.get_in_buffer());
    w.clear();
    w.push_back(&buf);
    w.push_back(&d); 
    c.set_input(w);
    d.set_output_buffer(c.get_in_buffer());
    
    a.skipfirstpop(true);  
    a.run(0); 
    b.run(1); 
    c.run(2); 
    d.run(3);
    d.wait();c.wait();b.wait();a.wait();
    printf("DONE\n");
    
    return 0;
}

