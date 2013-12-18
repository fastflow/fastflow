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
 *  Using ff_monode (multi-output node) as a pipeline stage. 
 *
 *    -----            ------            ------ 
 *   |  A  |--------> |  B   |--------> |  C   |
 *   |     |          |      |          |      |
 *    -----            ------            ------ 
 *        ^              |           
 *         \             |           
 *          -------------            
 *
 */

#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>  // for ff_minode ff_monode

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
    // this is pure virtual in ff_monode
    int set_output(std::vector<ff_node*>& w) {
        w = nodes;
        return 0;
    }

    void fill_nodes(std::vector<ff_node*>& w) { nodes = w;}

protected:
  std::vector<ff_node*> nodes;
};
/* --------------------------------------- */

class C: public ff_node {
public:
    void *svc(void *t) { 
        printf("C got %ld\n", (long)t);
        return GO_ON;
    }
};
/* --------------------------------------- */

int main() {
    A a;  B b;  C c;

    ff_pipeline pipe1;
    pipe1.add_stage(&a);
    pipe1.add_stage(&b);
    pipe1.wrap_around();

    std::vector<ff_node*> w;
    w.push_back(&a); w.push_back(&c);
    b.fill_nodes(w);

    ff_pipeline pipe;
    pipe.add_stage(&pipe1);
    pipe.add_stage(&c);
    pipe.run_and_wait_end();
    printf("DONE\n");
    
    return 0;
}

