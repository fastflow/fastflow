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
 * Date:   July 2014
 */
/*
 *    
 *  Testing the following skeleton:   pipeline(master-worker, multi-input);
 *
 *
 *                 ------------------------                  
 *               |              -----     |
 *               |        -----|     |    |
 *               v       |     |  W  |----
 *             -----     |      -----  -----------      -----
 *            |  E  |----|        .               |    |  N  |
 *            |     |    |        .    ---------- |--> |     |
 *             -----     |        .               |     -----
 *               ^       |      -----  -----------
 *               |        -----|  W  | ---
 *               |             |     |    |
 *               |              -----     |
 *                ------------------------    
 *
 *  2-stage pipeline, the first stage is a task-farm in a master-worker configuration 
 *  (workers send tasks back to the emitter thread), the second stage is a multi-input
 *  node getting input tasks from the workers (using an explicit instanciated channel)
 *
 *
 */      



#include <vector>
#include <iostream>
#include <ff/svector.hpp>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>
#include <ff/node.hpp>

#if !defined(HAS_CXX11_VARIADIC_TEMPLATES)
#error "this test requires the -DHAS_CXX11_VARIADIC_TEMPLATES compile flag"
#endif
  
using namespace ff;

struct W: ff_node {
    W():outbuffer(100,false) {}

    void *svc(void *task){
        printf("W(%d) got task %ld\n", get_my_id(), (long)task);
        outbuffer.put(task);
        return task;
    }

    void svc_end() {
        outbuffer.put(EOS);
    }

    void get_out_nodes(svector<ff_node*> &w) {
        w.push_back(&outbuffer);
    }

    ff_buffernode outbuffer;
    size_t output_channels;
};

// multi input node
struct N: ff_minode {
    void *svc(void *task){
        printf("N got task %ld\n",(long)task);
        return GO_ON;
    }
};

class E: public ff_node {
public:
    E(long numtasks):numtasks(numtasks) {}

    void *svc(void *task) {	
        if (task == NULL) {
            for(long i=1;i<=numtasks;++i)
                ff_send_out((void*)i);
            return GO_ON;
        }
        if (--numtasks <= 0) return NULL;
        return GO_ON;	
    }
private:
    long numtasks;
};



int main(int argc,  char * argv[]) {
    int nworkers=3;
    int streamlen=1000;
    if (argc>1) {
        if (argc<3) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nworkers streamlen\n";
            return -1;
        }        
        nworkers=atoi(argv[1]);
        streamlen=atoi(argv[2]);
    }
    if (nworkers<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    E emitter(streamlen);
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new W);
    ff_farm<> farm(w,&emitter);
    farm.remove_collector(); // this is needed here to avoid init errors!
    farm.wrap_around();

    ff_pipe<long> pipe(&farm,new N);
    if (pipe.run_and_wait_end()<0) return -1;
    printf("DONE\n");
    return 0;
}
