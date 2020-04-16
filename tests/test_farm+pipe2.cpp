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
 *  pipe(feedback(farm(pipe(stage1,stage2))), Gatherer)
 *
 *                    ---------------------------       
 *                   |                           |
 *                   |     --> (stage1->stage2)-- -  
 *       (farm)      v    |                        |
 *              Emitter --                         |--> Gatherer
 *                   ^    |                        |    (multi-input)
 *                   |     --> (stage1->stage2)-- -
 *                   |                           |
 *                   ---------------------------- 
 */
/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

class Stage1: public ff_monode_t<long> {
public:
    long * svc(long * t) {
        std::cout << "Stage1 got task " << *t << "\n";
        return t;
    }
};

class Stage2: public ff_monode_t<long> {   
public:
    long * svc(long * t) {
        std::cout << "Stage2 got task " << *t << "\n";
        bool odd = *t & 0x1;
        if (!odd) {
            long * nt = new long(*t);
            ff_send_out_to(nt, 1); // sends it forward
        }
        ff_send_out_to(t, 0); // sends it back        
        return GO_ON;
    }
}; 

struct Gatherer: ff_minode_t<long> {
    long* svc(long* t) {
        std::cout << "Gatherer, collected " << *t << " from " << get_channel_id() << "\n";
        delete t;
        return GO_ON;
    }
};

class Emitter: public ff_monode_t<long> {
public:
    long * svc(long * task) { 
        if (!task)  return new long(1000);

        std::cout << "Emitter task came back " << *task << " from " << get_channel_id() << "\n";
        (*task)--;
        
        if (*task<=0) { delete task; return EOS;}
        return task;
    }
};


int main(int argc, char * argv[]) {
    int nworkers  = 3;
    if (argc>1) {
        if (argc != 2) {
            std::cerr << "use:\n" << " " << argv[0] << " num-farm-workers\n";
            return -1;
        }
        nworkers  =atoi(argv[1]);
    }
    
    ff_farm farm;
    Emitter E;
    farm.add_emitter(&E); 

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) {
        // build worker pipeline 
        ff_pipeline * pipe = new ff_pipeline;
        pipe->add_stage(new Stage1, true);
        pipe->add_stage(new Stage2, true);
        w.push_back(pipe);
    }
    farm.add_workers(w);
    farm.wrap_around();
    farm.cleanup_workers();

    ff_pipeline pipe;
    pipe.add_stage(&farm);
    pipe.add_stage(new Gatherer, true);
    
    pipe.run();
    // wait all threads join
    if (pipe.wait()<0) {
        error("waiting pipeline\n");
        return -1;
    }

    std::cout << "DONE\n";
    return 0;
}
