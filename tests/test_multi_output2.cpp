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
 *         May 2015 (simplified)
 */
/*
 *    
 *  pipeline(master-worker, multi-input);
 *  (see test_multi_output5.cpp (second stage) for a different version having the farm collector)
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
 *  W: is a multi-output node
 *  N: is a multi-input node
 */      



#include <vector>
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

struct W: ff_monode_t<long> {
    long *svc(long *task){
        printf("W(%ld) got task %ld\n", get_my_id(), (long)task);
        ff_send_out_to(task, 0); // to the Emitter
        ff_send_out_to(task, 1); // to the next stage
        return GO_ON;
    }
};

// multi input node
struct N: ff_minode_t<long> {
    N(const long numtasks):numtasks(numtasks) {}
    long *svc(long *task){
        ticks_wait(10000); //for(volatile long i=0;i<10000; ++i);
        printf("N got task %ld\n",(long)task);
        ++received;
        return GO_ON;
    }
    void svc_end() {
        if (received != numtasks) abort();
    }
    
    long numtasks;
    long received = 0;
};

class E: public ff_node_t<long> {
public:
    E(long numtasks):numtasks(numtasks) {}

    int svc_init() {
        return 0;
    }    
    
    long *svc(long *task) {	
        if (task == NULL) {
            for(long i=1;i<=numtasks;++i)
                ff_send_out((long*)i);            
            return GO_ON;
        }
        printf("E: got back %ld numtasks=%ld\n", (long)task, numtasks);
        if (--numtasks <= 0) {
            printf("E sending EOS\n");
            return EOS;
        }
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
    ff_farm farm(w,&emitter); // this constructor adds the default collector !!!
    farm.remove_collector();
    farm.wrap_around();

    N multi_input(streamlen);
    ff_Pipe pipe(farm, multi_input);

    if (pipe.run_and_wait_end()<0) return -1;
    printf("DONE\n");
    return 0;
}
