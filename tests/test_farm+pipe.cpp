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
 *
 *                    ---------------------------       
 *                   |                           |
 *                   |     --> (stage1->stage2)--   
 *       (farm)      v    |              
 *              Emitter --  
 *                   ^    |               
 *                   |     --> (stage1->stage2)--
 *                   |                           |
 *                   ---------------------------- 
 */

#include <iostream>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>


using namespace ff;

class Stage1: public ff_node {
public:
    void * svc(void * task) {
        int *t = (int*)task;
        std::cout << "Stage1 got task " << *t << "\n";
        (*t)--;
        return task;
    }
};

class Stage2: public ff_node {
public:
    void * svc(void * task) {
        int *t = (int*)task;
        std::cout << "Stage2 got task " << *t << "\n";
        (*t)--;
        return task;
    }
}; 

class Emitter: public ff_node {
public:
    void * svc(void * task) { 
        if (task==NULL)  return new int(1000);

        int t = *(int*)task;
        if (t<=0) return NULL;
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

    ff_farm<> farm;
    Emitter E;
    farm.add_emitter(&E); 

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) {
        // build worker pipeline 
        ff_pipeline * pipe = new ff_pipeline;
        pipe->add_stage(new Stage1);
        pipe->add_stage(new Stage2);
        w.push_back(pipe);
    }
    farm.add_workers(w);
    farm.wrap_around();

    farm.run();
    // wait all threads join
    if (farm.wait()<0) {
        error("waiting farm freezing\n");
        return -1;
    }

    std::cout << "DONE\n";
    return 0;
}
