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

/* Some basic usage examples of the pipeline (and farm) pattern  */

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
    if (++k == 10) { printf("f1 END\n"); return NULL;}
    if (in==NULL) {
        //int *V = new int[k+5]{}; brace-init does not work with Intel compiler 
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

typedef std::function<myTask*(myTask*,ff_node*const)> func_t;

int main() {
    /* ------------------------------------------- */
    // Basic 3-stage pipeline f1;f2;f3
    ff_pipe<myTask> pipe1(f1,f2,f3);
    pipe1.run_and_wait_end();
    printf("done 1st\n\n");
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    {
        // functions may also be lambda functions
        auto lambda1 = [] (myTask *in,ff_node*const) -> myTask* {
            return f1(in,nullptr);
        };
        ff_pipe<myTask> pipe1(lambda1,f2,f3);
        pipe1.run_and_wait_end();
        printf("done 1st with lambda\n\n");
    }
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    // ...with some more stages
    ff_pipe<myTask> pipe2(f1,f2,f2,f2,f3);
    pipe2.run_and_wait_end();
    printf("done 2nd\n\n");
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    // composing 2 pipelines
    ff_pipe<myTask> pipe0(f2,f2);
    ff_pipe<myTask> pipe3(f1, &pipe0,f3);
    pipe3.run_and_wait_end();
    printf("done 3rd\n\n");
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    // farm introduction. The pipeline has also a feedback channel.
    ff_pipe<myTask> pipe4(f1,new ff_farm<>((func_t)f2,3),new ff_farm<>((func_t)f3,2));
    pipe4.add_feedback();
    pipe4.run_and_wait_end();
    printf("done 4th\n\n");
    /* ------------------------------------------- */

    /* ------------------------------------------- */
    // ... just a little bit more complicated. 
    // A pipeline with 3 farms, each farm without collector.
    // The emitter of the first farm produces the stream.
    auto lambda = []() -> void* {
        static int k = 0;
        if (k++ == 10) { printf("Emitter END\n"); return NULL;}
        int *V = new int[k+5];
        for(int i=0;i<(k+5);++i) V[i]=0;
        return new myTask(k+5, V);
    };
    struct Emitter:public ff_node {
        std::function<void*()> F;
        Emitter(std::function<void*()> F):F(F) {}
        void *svc(void*) { return F(); }
    };
    ff_farm<> farm1((func_t)f1,2);
    ff_farm<> farm2((func_t)f2,3);
    ff_farm<> farm3((func_t)f3,2);
    ff_pipe<myTask> pipe5(&farm1,&farm2,&farm3);
    farm1.add_emitter(new Emitter(lambda));
    farm1.remove_collector();
    farm2.setMultiInput();
    farm2.remove_collector();
    farm3.setMultiInput();
    farm3.remove_collector();
    pipe5.run_and_wait_end();
    printf("done 5th\n\n");
    /* ------------------------------------------- */

    k=-1; // reset k

    /* ------------------------------------------- */
    // Pipeline of 3 stages: sequential, sequential     
    // and farm-with-feedback.
    ff_farm<> farm0((func_t)f3, 3);
    ff_pipe<myTask> pipe6(f1,f2,&farm0);
    struct Scheduler:public ff_node {
        ff_loadbalancer* lb;
        Scheduler(ff_loadbalancer* lb):lb(lb) {}
        void *svc(void *t) {
            // do something smart here :)
            if (lb->get_channel_id() == -1) {
                return t;
            }
            return GO_ON;
        }            
        void eosnotify(int id) {
            if (id==-1) lb->broadcast_task(EOS);
        }
    };
    farm0.add_emitter(new Scheduler(farm0.getlb()));
    farm0.remove_collector();
    farm0.wrap_around();
    pipe6.run_and_wait_end();
    printf("done 6th\n\n");
    /* ------------------------------------------- */


    return 0;
}
