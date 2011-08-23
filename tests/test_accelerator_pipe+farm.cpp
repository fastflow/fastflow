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
 * Mixing FastFlow farm with pipeline workers in the accelerator mode
 *
 *                  |<------ 3-stages pipeline ------>|
 *        .
 *        .                         |stage2|   
 *        .                         |      | 
 *    main-flow ---->stage1 -->farm |stage2|---> stage3
 *        .                         |      |       |
 *        .                         |stage2|       |
 *        .                                        |
 *    main-flow <----------------------------------         
 *
 *
 */

#include <stdlib.h>
#include <iostream>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>

using namespace ff;

enum { IN_QUEUE_SIZE=1024, OUT_QUEUE_SIZE=2048};

// define your task
typedef int my_task_t;

class Stage1: public ff_node {
public:
    void * svc(void * task) {
        std::cout << "Stage1 got task\n";
        return task;
    }
};

class Stage2: public ff_node {
public:
    void * svc(void * task) {
        std::cout << "Stage2 got task\n";
        return task;
    }
}; 

class Stage3: public ff_node {
public:
    void * svc(void * task) {
        std::cout << "Stage3 got task\n";
        return task;
    }
}; 


int main(int argc, char * argv[]) {
    void * result = NULL;

    int mstreamlen= 0;
    int nworkers  = 0;
    int iterations= 0;

    if (argc != 4) {
        std::cerr << "use:\n" << " " << argv[0] << " max-stream-length num-farm-workers iterations\n";
        return -1;
    }
    mstreamlen=atoi(argv[1]);
    nworkers  =atoi(argv[2]);
    iterations=atoi(argv[3]);

    srandom(::getpid()+(getusec()%4999)); // init seed
     
    ff_farm<> farm;
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) {
        w.push_back(new Stage2);
    }
    farm.add_workers(w);
    farm.add_collector(NULL); // we just want a generic gather without any user filter

    ff_pipeline pipe(true);
    pipe.add_stage(new Stage1);
    pipe.add_stage(&farm);
    pipe.add_stage(new Stage3);

    for(int i=0;i<iterations;++i) {

        if (pipe.run_then_freeze()<0) {
            error("running pipe\n");
            return -1;
        }
        

        for(int j=0;j<mstreamlen;++j) {
            my_task_t * task = new my_task_t(i+j);

            if (!pipe.offload(task)) {
                error("offloading task\n");
                return -1;
            }

            if (pipe.load_result_nb(&result)) {
                std::cout << "result= " << *((int*)result) << "\n";
                delete ((int*)result);
            }            
        }

        // offload End-Of-Stream
        if (!pipe.offload((void *)FF_EOS)) {
            error("offloading EOS\n");
            return -1;
        }

        // asynchronously wait results
        do {
            if (pipe.load_result_nb(&result)) {
                if (result==(void*)FF_EOS) break;

                /* do something useful, probably using onother thread */

                std::cout << "result= " << *((int*)result) << "\n";
                delete ((int*)result);
            } 

            /* do something else */
            
        } while(1);

        std::cout << "got all results iteration= " << i << "\n";

        // here join
        if (pipe.wait_freezing()<0) {
            error("waiting farm freezing\n");
            return -1;
        }
    }

    // wait all threads join
    if (pipe.wait()<0) {
        error("error waiting farm\n");
        return -1;
    }

    std::cerr << "DONE, time= " << pipe.ffTime() << " (ms)\n";
    farm.ffStats(std::cout);

    return 0;
}
