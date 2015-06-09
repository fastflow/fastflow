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

/**
 * \file pipe_basic.cpp
 * \ingroup applications
 * \brief Some basic usage examples of the pipeline (and farm) pattern 
 *
 * @include pipe_basic.cpp
 */
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>


using namespace ff;

// a stream counter
static int k=-1;

// a task
struct myTask {
    myTask(int n,int* V):n(n),V(V) {}
    int n;
    int *V;
};

// f1 f2 and f3 are 3 functions which get in input a task
// pointer and return a task pointer.
// memory management is up to the user

myTask* f1(myTask* in, ff_node*const) {
    if (++k == 10) { 
        printf("f1 END\n"); 
        if (in) {delete [] in->V; delete in;} 
        return (myTask*)EOS;
    }
    if (in==nullptr) {
        int *V = new int[k+5];
        for(int i=0;i<(k+5);++i) V[i]=0;
        return new myTask(k+5, V);
    }
    return in;
}
myTask* f2(myTask *in, ff_node*const) {
    for(int i=0;i<in->n;++i) in->V[i]++;
    return in;
}
myTask* f3(myTask *in, ff_node*const) {
    printf("f3 received: ");
    for(int i=0;i<in->n;++i) printf(" %d ", in->V[i]);
    printf("\n");
    return in;
}

struct Deleter:ff_node_t<myTask> {
    myTask *svc(myTask *task) { delete [] task->V; delete task; return GO_ON;}
};

int main() {
    /* ------------------------------------------- */
    {
        // Basic 3-stage pipeline f1;f2;f3 (plus a final stage for deleting tasks)
        // Task deletion can be embedded in the f3 function but then it cannot be 
        // used for the next examples
        ff_node_F<myTask,myTask> F1(f1), F2(f2), F3(f3);
        ff_Pipe<> pipe1(F1,F2,F3, make_unique<Deleter>());
        pipe1.run_and_wait_end();
        printf("done 1st\n\n");
    }
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    {
        // functions may also be lambda functions
        auto lambda1 = [] (myTask *in,ff_node*const) -> myTask* {
            return f1(in,nullptr);
        };
        ff_node_F<myTask,myTask> L1(lambda1), F2(f2), F3(f3);

        ff_Pipe<> pipe1(L1,F2,F3, make_unique<Deleter>());
        pipe1.run_and_wait_end();
        printf("done 1st with lambda\n\n");
    }
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    {
        // ...with some more stages
        ff_node_F<myTask,myTask> F1(f1), F21(f2), F22(f2), F23(f2), F3(f3);
        ff_Pipe<> pipe2(F1,F21,F22,F23,F3, make_unique<Deleter>());
        pipe2.run_and_wait_end();
        printf("done 2nd\n\n");
    }
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    {
        // composing 2 pipelines
        ff_node_F<myTask> F1(f1), F21(f2), F22(f2), F3(f3); 
        ff_Pipe<myTask> pipe0(F21,F22);
        ff_Pipe<>       pipe3(F1, pipe0,F3, make_unique<Deleter>());
        pipe3.run_and_wait_end();
        printf("done 3rd\n\n");
    }
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    {
        // farm introduction. The pipeline has also a feedback channel.
        ff_node_F<myTask> F1(f1);
        ff_Farm<myTask> farm1(f2,3);
        ff_Farm<myTask> farm2(f3,2);
        ff_Pipe<> pipe4(F1,farm1, farm2);
        pipe4.wrap_around();
        pipe4.run_and_wait_end();
        printf("done 4th\n\n");
    }
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    {
        // ... just a bit more complex.
        // A pipeline with 3 farms, each farm without collector.
        // The emitter of the first farm produces the stream.
        auto lambda = []() -> myTask* {
            static int k = 0;
            if (k++ == 10) { 
                printf("Emitter END\n"); 
                return (myTask*)EOS;
            }
            int *V = new int[k+5];
            for(int i=0;i<(k+5);++i) V[i]=0;
            return new myTask(k+5, V);
        };

        struct Emitter:public ff_node_t<myTask> {
            std::function<myTask*()> F;
            Emitter(std::function<myTask*()> F):F(F) {}
            myTask *svc(myTask*) { return F(); }
        };
        struct Collector:public ff_node_t<myTask> {
            myTask *svc(myTask *task) { delete [] task->V; delete task; return GO_ON;}
        };
        ff_Farm<myTask> farm1(f1,2);
        ff_Farm<myTask> farm2(f2,3);
        ff_Farm<myTask> farm3(f3,2);
        ff_Pipe<myTask> pipe5(farm1,farm2,farm3);
        Emitter E(lambda);
        farm1.add_emitter(E);
        farm1.remove_collector();
        farm2.setMultiInput();
        farm2.remove_collector();
        farm3.setMultiInput();
        farm3.remove_collector();
        Collector C;
        farm3.add_collector(C);
        pipe5.run_and_wait_end();
        printf("done 5th\n\n");
    }
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    {
        // Pipeline of 3 stages: sequential, sequential     
        // and farm-with-feedback.
        ff_Farm<myTask> farm0(f3, 3);
        ff_node_F<myTask> F1(f1), F2(f2);
        ff_Pipe<myTask> pipe6(F1,F2,farm0);
        struct Scheduler:public ff_node_t<myTask> {
            ff_loadbalancer* lb;
            Scheduler(ff_loadbalancer* lb):lb(lb) {}
            myTask *svc(myTask *t) {
                // do something smart here :)
                if (lb->get_channel_id() == -1) {
                    return t;
                }
                delete [] t->V; delete t;
                return GO_ON;
            }            
            void eosnotify(ssize_t id) {
                if (id==-1) lb->broadcast_task(EOS);
            }
        };
        Scheduler S(farm0.getlb());
        farm0.add_emitter(S);
        farm0.remove_collector();
        farm0.wrap_around();
        pipe6.run_and_wait_end();
        printf("done 6th\n\n");
    }

    /* ------------------------------------------- */
    return 0;
}
