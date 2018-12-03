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
 *       2-stages pipeline
 *               
 *          -------------------------------- 
 *         |                                |
 *         |                     ---> FU -->| 
 *         v                    |           | 
 *        MU ----> Scheduler --- ---> FU -->| 
 *         ^                    |           |
 *         |                     ---> FU -->|
 *         |                                |
 *          --------------------------------
 *
 */

#include <vector>
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

/*
 * NOTE: this is a multi-input node
 */
class MU: public ff_minode {
public:
    MU(int numtasks):
        numtasks(numtasks), k(0) {}

    void* svc(void* task) {
        if (task==NULL) {
            printf("MU starting producing tasks\n");
            for(long i=1;i<=numtasks;++i)
                ff_send_out((void*)i);
            return GO_ON;
        }

        long t = (long)task;
        if (--t > 0) ff_send_out((void*)t);
        else if (++k == numtasks) return EOS;
        return GO_ON;
    }
private:
    long numtasks;
    long k;
};

struct Scheduler: public ff_node {
    void* svc(void* task) {        
        return task;
    }
};

struct FU: public ff_node {
    void* svc(void* task) {
        printf("FU (%ld) got one task (%ld)\n", get_my_id(), (long)task);
        return task;
    }
};


int main(int argc, char* argv[]) {
    int nw=3;
    int numtasks=1000;
    if (argc>1) {
        if (argc < 3) {
            std::cerr << "use:\n" << " " << argv[0] << " numworkers ntasks\n";
            return -1;
        }
        nw=atoi(argv[1]);
        numtasks=atoi(argv[2]); 
    }

    ff_pipeline pipe;
    ff_farm     farm;

    std::vector<ff_node *> w;
    for(int i=0;i<nw;++i) 
        w.push_back(new FU);
    farm.add_emitter(new Scheduler);
    farm.add_workers(w);

    pipe.add_stage(new MU(numtasks));
    pipe.add_stage(&farm);
    
    /* this is needed to allow the creation of output buffer in the 
     * farm workers 
     */
    farm.remove_collector();

    pipe.wrap_around();    
    pipe.run_and_wait_end();

    return 0;
}
