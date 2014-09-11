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

/* Some basic usage examples of the ff_farm  */

#include <ff/farm.hpp>

using namespace ff;

// just a task
struct ff_task {};

// just a function
ff_task *F(ff_task *t, ff_node*const node) {
    printf("hello I've got one task my id is=%d\n", node->get_my_id());
    return t;
}

// just an ff_node
struct seq: ff_node {
    void *svc(void *t) {
	printf("seq %d got one task\n", get_my_id());
	return t;
    }
};

int main() {    
    /* ------------------------------------------- */
    // First version using functional replication
    // NOTE: no data stream in input to the farm
    ff_farm<> farm((std::function<ff_task*(ff_task*,ff_node*const)>)F, 5);
    farm.run_and_wait_end();    
    printf("done 1st\n\n");
    /* ------------------------------------------- */

    /* ------------------------------------------- */    
    // As in the previous case but the farm is set up
    // as a software accelerator. The stream of tasks
    // is generated from the main.
    ff_farm<> farmA((std::function<ff_task*(ff_task*,ff_node*const)>)F, 5,true);
    farmA.run();
    for(int i=0;i<10;++i) farmA.offload(new long(i));
    farmA.offload(EOS);
    farmA.wait();
    printf("done 2nd\n\n");
    /* ------------------------------------------- */

    /* ------------------------------------------- */
    // Farm using a pool of ff_node and a skeleton 
    // without the Collector
    // NOTE: no data stream in input to the farm
    std::vector<ff_node*> W;
    for(int i=0;i<4;++i) W.push_back(new seq);
    ff_farm<> farm_wo_collector(W);
    farm_wo_collector.remove_collector(); 
    farm_wo_collector.run_and_wait_end();
    printf("done 3nd\n\n");
    /* ------------------------------------------- */

#if defined(HAS_CXX11_LAMBDA)
    /* ------------------------------------------- */
    // Version using a lambda function to define the Emitter thread,
    // which generates the stream. No collector present.
#if 0
    ff_farm<> farm2;
    const int K = 10;
    auto lambda = [K]() -> void* {
        static int k = 0;
        if (k++ == K) return NULL;
        return new int(k);
    };
    struct Emitter:public ff_node {
        std::function<void*()> F;
        Emitter(std::function<void*()> F):F(F) {}
        void *svc(void*) { return F(); }
    };
    farm2.add_emitter(new Emitter(lambda));
    W.clear();
    for(int i=0;i<4;++i) W.push_back(new seq);
    farm2.add_workers(W);
    farm2.run_and_wait_end();
#else
    const int K = 10;
    int k = 0;
    auto lambda = [K,&k]() -> void* {
        if (k++ == K) return NULL;
        return new int(k);
    };
    auto myF=[](ff_task *t,ff_node*const node)->ff_task*{ 
        printf("hello I've got one task my id is=%d\n", node->get_my_id());
        return t;
    };
    struct Emitter:public ff_node {
        std::function<void*()> F;
        Emitter(std::function<void*()> F):F(F) {}
        void *svc(void*) { return F(); }
    };
    ff_farm<> farm2((std::function<ff_task*(ff_task*,ff_node*const)>)myF,4);

    farm2.remove_collector();
    farm2.add_emitter(new Emitter(lambda));
    farm2.run_and_wait_end();
#endif
    printf("done 4th\n\n");
    /* ------------------------------------------- */

    /* ------------------------------------------- */
    // Another version using a lambda function to define the Emitter thread,
    // which generates the stream. No collector present.

    // just an ff_node
    struct MyNode: ff_node {
        void *svc(void *t) {
            printf("worker %d got one task\n", get_my_id());
            return t;
        }
    };
    const int K2 = 20;
    auto lambda2 = [K2]() -> void* {
        static int k = 0;
        if (k++ == K2) return NULL;
        return new int(k);
    };
    W.clear();
    for(int i=0;i<7;++i) W.push_back(new MyNode);
    ff_farm<> farm3(W, new Emitter(lambda2));
    farm3.run_and_wait_end();
    printf("done 5th\n\n");
#endif 

    return 0;
}


