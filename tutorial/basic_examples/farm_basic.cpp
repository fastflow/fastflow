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
 * Date  : 2014
 *         
 */

/* Some basic usage examples of the ff_farm  */

#include <ff/farm.hpp>

using namespace ff;

// just a task
struct ff_task {};

// just a function
ff_task *F(ff_task *t, ff_node*const node) {
    printf("hello I've got one task my id is=%ld\n", node->get_my_id());
    return t;
}

// just an ff_node
struct seq: ff_node_t<ff_task> {
    ff_task *svc(ff_task *t) {
        printf("seq %ld got one task\n", get_my_id());
        return t;
    }
};


int main() {    
    /* ------------------------------------------- */
    // First version using functional replication
    // NOTE: no data stream in input to the farm
    ff_Farm<ff_task> farm(F, 5);
    farm.run_and_wait_end();    
    printf("done 1st\n\n");
    /* ------------------------------------------- */

    /* ------------------------------------------- */    
    // As in the previous case but the farm is set up
    // as a software accelerator. The stream of tasks
    // is generated from the main.
    ff_Farm<ff_task> farmA(F, 5,true);
    farmA.run();
    for(long i=1;i<=10;++i) farmA.offload((ff_task*)i);
    farmA.offload(EOS);
    farmA.wait();
    printf("done 2nd\n\n");
    /* ------------------------------------------- */

    /* ------------------------------------------- */
    // Farm using a pool of ff_node and a skeleton 
    // without the Collector
    // NOTE: no data stream in input to the farm

#if 1  // first option for building an "external" vector of unique pointers
       // then moving the vector to the farm
    std::vector<std::unique_ptr<ff_node> > W;
    for(int i=0;i<4;++i) W.push_back(make_unique<seq>()); 
    ff_Farm<> farm_wo_collector(std::move(W));
#else  // second option, using lambdas
    ff_Farm<> farm_wo_collector( []() { 
            std::vector<std::unique_ptr<ff_node> > W; 
            for(int i=0;i<4;++i) W.push_back(make_unique<seq>()); 
            return W; 
        }() );
#endif
    farm_wo_collector.remove_collector(); 
    farm_wo_collector.run_and_wait_end();
    printf("done 3nd\n\n");
    /* ------------------------------------------- */

    /* ------------------------------------------- */
    // Version using a lambda function to define the Emitter thread,
    // which generates the stream. No collector present.
    const int K = 10;
    int k = 0;
    auto lambda = [K,&k]() -> long* {
        if (k++ == K) return NULL;
        return new long(k);
    };
    auto myF=[](long *t,ff_node*const node)->long*{ 
        printf("hello I've got one task my id is=%ld\n", node->get_my_id());
        delete t;
        return t;
    };
    struct Emitter:public ff_node_t<long> {
        std::function<long*()> F;
        Emitter(std::function<long*()> F):F(F) {}
        long *svc(long*) { return F(); }
    };
    ff_Farm<long> farm2(myF,4);

    farm2.remove_collector();
    Emitter E(lambda);
    farm2.add_emitter(E);
    farm2.run_and_wait_end();
    printf("done 4th\n\n");
    /* ------------------------------------------- */
    
    /* ------------------------------------------- */
    // Another version using a lambda function to define the Emitter thread,
    // which generates the stream. No collector present.

    // just an ff_node
    struct MyNode: ff_node_t<ff_task> {
        ff_task *svc(ff_task *t) {
            printf("worker %ld got one task\n", get_my_id());
            delete t;
            return t;
        }
    };
    const int K2 = 20;
    auto lambda2 = [K2]() -> ff_task* {
        static int k = 0;
        if (k++ == K2) return NULL;
        return new ff_task;
    };
    struct Emitter2:public ff_node_t<long, ff_task> {
        std::function<ff_task*()> F;
        Emitter2(std::function<ff_task*()> F):F(F) {}
        ff_task *svc(long*) { return F(); }
    };

    Emitter2 E2(lambda2);
    ff_Farm<long,ff_task> farm3([]() {     
            std::vector<std::unique_ptr<ff_node> > W;
            for(int i=0;i<2;++i) W.push_back(make_unique<MyNode>());
            return W;
        }() );
    farm3.add_emitter(E2);
    farm3.remove_collector();
    farm3.run_and_wait_end();
    printf("done 5th\n\n");

    return 0;
}


