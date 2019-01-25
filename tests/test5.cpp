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
 * This program tests composition of pipeline modules.
 *
 *
 *     |---------- 3-stage pipeline -----------|
 *     |                   |                   | 
 *     v                   v                   v
 *    stage1-->pipe (stage2_1->stage2_2)|-->stage2
 *                   ^                ^
 *                   |- 2-stage pipe -|
 */

#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

#if defined(USE_FF_ALLOCATOR)
#define MAKE_VALGRIND_HAPPY 1
#include <ff/allocator.hpp>
static ff_allocator ffalloc;
#define MALLOC(X) ffalloc.malloc(X)
#define FREE(X)   ffalloc.free(X)
#else
#define MALLOC(X) ::malloc(X)
#define FREE(X)   ::free(X)
#endif

class Stage1: public ff_node {
public:
    Stage1(unsigned int streamlen):streamlen(streamlen),cnt(0){}

    void * svc(void *) {
        int * t;
        t = (int*)MALLOC(sizeof(int));
        if (!t) abort();
        
        *t=cnt++;
        if (cnt > streamlen) {
            FREE(t);
            t = NULL; // EOS
        }
        return t;
    }
#if defined(USE_FF_ALLOCATOR)
    int   svc_init() { 
        if (ffalloc.registerAllocator()<0) {
            error("registerAllocator fails\n");
            return -1;
        }
        return 0; 
    }
#endif

private:
    unsigned int streamlen;
    unsigned int cnt;
};


class Stage2_1: public ff_node {
public:
    void * svc(void * task) {
        return task;
    }
};

class Stage2_2: public ff_node {
public:
    void * svc(void * task) {
        return task;
    }
};

class Stage3: public ff_node {
public:
    Stage3():sum(0){}

    void * svc(void * task) {
        int * t = (int *)task;
        if (!t)  abort();
        sum +=*t;
        FREE(task);
        task = GO_ON; // we want to be sure to continue
        return task;
    }
#if defined(USE_FF_ALLOCATOR)
    int   svc_init() { 
        if (ffalloc.register4free()<0) {
            error("register4free fails\n");
            return -1;
        }
        
        return 0; 
    }
#endif
    void  svc_end() {
        std::cout << "Sum: " << sum << "\n";
    }

private:
    unsigned int sum;
};



int main(int argc, char * argv[]) {
    int streamlen=1000;
    if (argc>1) {
        if (argc!=2) {
            std::cerr << "use: "  << argv[0] << " streamlen\n";
            return -1;
        }
        streamlen = atoi(argv[1]);
    }

#if defined(USE_FF_ALLOCATOR)
    // init allocator
    ffalloc.init();
#endif

#if 1 // first version using cleanup_nodes 
    ff_pipeline pipe;
    pipe.add_stage(new Stage1(streamlen));

    ff_pipeline *pipe2=new ff_pipeline;
    pipe2->cleanup_nodes();  // cleanup at exit
    pipe2->add_stage(new Stage2_1);
    pipe2->add_stage(new Stage2_2);

    // add the farm module as a second pipeline stage
    pipe.add_stage(pipe2);

    // add last stage to the main pipeline
    pipe.add_stage(new Stage3);

    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    pipe.cleanup_nodes(); // cleanup at exit
#else // second version allocating stages on the stack 

    // build main pipeline
    ff_pipeline pipe2; // must be before pipe, because of destruction sequence
    ff_pipeline pipe;

    Stage1 s1(streamlen);
    pipe.add_stage(&s1);

    Stage2_1 s21;
    Stage2_2 s22;
    pipe2.add_stage(&s21);
    pipe2.add_stage(&s22);

    // add the farm module as a second pipeline stage
    pipe.add_stage(&pipe2);

    // add last stage to the main pipeline
    Stage3 s3;
    pipe.add_stage(&s3);

    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    // no cleanup needed here
#endif
    pipe.ffStats(std::cout);

    return 0;
}
