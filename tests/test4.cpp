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
 * Mixing FastFlow pipeline and farm. The farm module has neihter the Emitter
 * nor the Collector filters.
 *
 *     |---------- 3-stage pipeline -----------|
 *     |                   |                   | 
 *     |                   v                   |
 *     |            |(stage2_1->stage2_2)|     |
 *     v            |                    |     v
 *    stage1-->farm |(stage2_1->stage2_2)|-->stage3
 *                  |                    |
 *                  |(stage2_1->stage2_2)|
 *                    ^                ^
 *                    |- 2-stage pipe -|
 */

#include <iostream>
#include <ff/ff.hpp>
#include <ff/allocator.hpp>


using namespace ff;

static ff_allocator ffalloc;

class Stage1: public ff_node {
public:
    Stage1(unsigned int streamlen):streamlen(streamlen),cnt(0){}

    void * svc(void *) {
        int * t;
        t = (int*)ffalloc.malloc(sizeof(int));
        if (!t) abort();
        
        *t=cnt++;
        if (cnt > streamlen) {
            ffalloc.free(t);
            t = NULL; // EOS
        }
        return t;
    }

    int   svc_init() { 
        if (ffalloc.registerAllocator()<0) {
            error("registerAllocator fails\n");
            return -1;
        }
        return 0; 
    }

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
        ffalloc.free(task);
        task = GO_ON; // we want to be sure to continue
        return task;
    }

    int   svc_init() { 
        if (ffalloc.register4free()<0) {
            error("register4free fails\n");
            return -1;
        }
        
        return 0; 
    }
    void  svc_end() {
        std::cout << "Sum: " << sum << "\n";
    }

private:
    unsigned int sum;
};



int main(int argc, char * argv[]) {
    int nworkers=3;
    int streamlen=1000;


    if (argc>1) {
        if (argc!=3) {
            std::cerr << "use: "  << argv[0] << " streamlen num-farm-workers\n";
            return -1;
        }

        streamlen=atoi(argv[1]);
        nworkers=atoi(argv[2]);    
    }

	std::cerr << "Init allocator ...";
    // init allocator
    ffalloc.init();
	std::cerr << " Done\n";
    // bild main pipeline
    ff_pipeline pipe;
    pipe.add_stage(new Stage1(streamlen));

    // build the farm module without the Emitter and the Collector filters
    ff_farm farm;
    // we just want a generic gather without any user filter
    farm.add_collector(NULL); 
   
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) {
        // build worker pipeline 
        ff_pipeline * pipe2 = new ff_pipeline;
        pipe2->add_stage(new Stage2_1);
        pipe2->add_stage(new Stage2_2);
        w.push_back(pipe2);
    }
    farm.add_workers(w);

    // add the farm module as a second pipeline stage
    pipe.add_stage(&farm);

    // add last stage to the main pipeline
    pipe.add_stage(new Stage3);
    std::cerr << "Starting ...\n";
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }

    pipe.ffStats(std::cout);
    return 0;
}
