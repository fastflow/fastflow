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
 *          -------------------------------- <--
 *         |                                |   |
 *         |                     ---> FU ---    |
 *         v                    |               |
 *        MU ----> Scheduler --- ---> FU ------- 
 *         ^                    |
 *         |                     ---> FU ---
 *         |                                |
 *          --------------------------------
 *
 */

#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>
#include <ff/node.hpp>

using namespace ff;

/*
 * NOTE: this is a multi-input node ff_minode !!
 */
class MU: public ff_minode {
public:
    MU(int numtasks, std::vector<ff_node*> & w):
        numtasks(numtasks), k(0), workers(w) {}

    void* svc(void* task) {
        if (task==NULL) {
            for(long i=1;i<=numtasks;++i)
                ff_send_out((void*)i);
            return GO_ON;
        }

        long t = (long)task;
        if (--t > 0) ff_send_out((void*)t);
        else if (++k == numtasks) return NULL;
        return GO_ON;
    }

    /* this allows to add the input channels by the user */
    int set_input(std::vector<ff_node*>& w) {
        printf("called set_input\n");
        w = workers;
        return 0;
    }
private:
    long numtasks;
    long k;
    std::vector<ff_node*> workers;
};

class Scheduler: public ff_node {
public:
    void* svc(void* task) {


        
        return task;
    }
};


class FU: public ff_node {
public:
    void* svc(void* task) {
	printf("FU (%d) got one task\n", get_my_id());
	return task;
    }
};



int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "use:\n" << " " << argv[0] << " numworkers ntasks\n";
        return -1;
    }
    int nw=atoi(argv[1]);
    int numtasks=atoi(argv[2]); 

    ff_pipeline pipe;
    ff_farm<>   farm;
    std::vector<ff_node *> w;
    for(int i=0;i<nw;++i) 
	w.push_back(new FU);
    farm.add_emitter(new Scheduler);
    farm.add_workers(w);
    pipe.add_stage(new MU(numtasks, w));
    pipe.add_stage(&farm);

    /* -------------------- */
    farm.remove_collector();
    pipe.wrap_around();
    /* -------------------- */

    pipe.run_and_wait_end();

    return 0;
}
