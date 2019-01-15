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
 * Mixing FastFlow pipeline with farm stages in the accelerator mode
 *
 *
 *                        |(stage1)|        |(stage2)|   
 *                        |        |        |        |  
 *    main-flow ---->farm |(stage1)|---farm |(stage2)|--------   
 *        .               |        |        |        |       | 
 *        .               |(stage1)|        |(stage2)|       | 
 *        .                                                  |
 *    main-flow <---------------------------------------------         
 *
 *
 */

#include <stdlib.h>
#include <iostream>
#include <ff/ff.hpp>
#include <ff/mapping_utils.hpp>


#if __linux__
#include <sys/time.h>
#include <sys/resource.h>
#include <asm/unistd.h>
#define gettid() syscall(__NR_gettid)
#else
#define gettid() 0
#endif

using namespace ff;

enum { IN_QUEUE_SIZE=1024, OUT_QUEUE_SIZE=2048};

// define your task
typedef int my_task_t;

class Stage1: public ff_node {
public:
    Stage1(int id):num_tasks(0), id(id){}

    void * svc(void * task) {
        ++num_tasks;
        return task;
    }

    void svc_end() {
        std::cout << "Stage1-Worker" << id << " Received " << num_tasks << " tasks." << std::endl;
    }
private:
    int num_tasks;
    int id;
};

class Stage2: public ff_node {
public:
    Stage2(int id):num_tasks(0), id(id){}

    void * svc(void * task) {
        ++num_tasks;
        return task;
    }

    void svc_end() {
        std::cout << "Stage2-Worker" << id << " Received " << num_tasks << " tasks." << std::endl;
    }
private:
    int num_tasks;
    int id;
};
class Emitter: public ff_node {
public:
    void * svc(void * task) { return task;}
};

class Collector: public ff_node {
public:
    void * svc(void * task) { return task;}

};



int main(int argc, char * argv[]) {
    void * result = NULL;

    int streamlen= 1000;
    int nworkers  = 3;
    if (argc>1) {
        if (argc != 3) {
            std::cerr << "use:\n" << " " << argv[0] << " stream-length num-farm-workers\n";
            return -1;
        }
        streamlen=atoi(argv[1]);
        nworkers  =atoi(argv[2]);
    }

    ff_farm farm_1(false, IN_QUEUE_SIZE, OUT_QUEUE_SIZE);
    Emitter E;
    Collector C;
    farm_1.add_emitter(&E); 
    farm_1.add_collector(&C);
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) {
        w.push_back(new Stage1(i));
    }
    farm_1.add_workers(w);

    ff_farm farm_2(false, IN_QUEUE_SIZE, OUT_QUEUE_SIZE);
    Emitter E2;
    Collector C2;
    farm_2.add_emitter(&E2); 
    farm_2.add_collector(&C2);
    std::vector<ff_node *> w2;
    for(int i=0;i<nworkers;++i) {
        w2.push_back(new Stage2(i));
    }
    farm_2.add_workers(w2);

    ff_pipeline * pipe = new ff_pipeline(true);
    pipe->add_stage(&farm_1);
    pipe->add_stage(&farm_2);


    /* ---------------------------------------------------- */
      
    if (pipe->run_then_freeze()<0) {
        error("running pipe\n");
        return -1;
    }
        
    int received_results=0;

    for(int j=0;j<streamlen;++j) {
        my_task_t * task = new my_task_t(j);

        if (!pipe->offload(task)) {
            error("offloading task\n");
            return -1;
        }

        // Try to get results, if there are any
        // If there aren't any deadlock problems, the following piece 
        // of code can be moved outside the for-j loop using 
        // the synchronous load_result method.
        if (pipe->load_result_nb(&result)) {
            ++received_results;

            /* do something useful, probably using another thread */

            delete ((int*)result);
        }            
    }

    // offload End-Of-Stream
    if (!pipe->offload((void *)FF_EOS)) {
        error("offloading EOS\n");
        return -1;
    }

    // asynchronously wait results
    do {
        if (pipe->load_result_nb(&result)) {
            if (result==(void*)FF_EOS) break;
            /* do something useful, probably using onother thread */

            ++received_results;
            delete ((int*)result);
        } 

        /* do something else */
            
    } while(1);

    // here join
    if (pipe->wait_freezing()<0) {
        error("waiting pipe freezing\n");
        return -1;
    }
    
    printf("Received %u results.\n", received_results);


    // wait all threads join
    if (pipe->wait()<0) {
        error("waiting pipe freezing\n");
        return -1;
    }

    std::cerr << "DONE, time= " << pipe->ffTime() << " (ms)\n";
    pipe->ffStats(std::cout);

    return 0;
}
