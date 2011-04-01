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
 * Very basic test for the FastFlow pipeline (3 stages).
 *
 */

#include <iostream>
#include <ff/pipeline.hpp>

using namespace ff;


// generic stage
class Stage1: public ff_node {
public:
    Stage1(unsigned int streamlen):streamlen(streamlen)  {}

    void * svc(void * task) {
        unsigned int * t = (unsigned int *)task;
        
        if (streamlen > 0) {
            t = (unsigned int*)malloc(sizeof(int));          
            *t=streamlen--;
            task = t;
            return task;
        } else
            return NULL;
    }
    void  svc_end() {
        std::cout << "Stage1 completed: " << ff_node::get_my_id() << "\n";
    }

private:
    unsigned int streamlen;
};

class Stage2: public ff_node {
public:
    void * svc(void * task) {
        unsigned int * t = (unsigned int *)task;
        
        if (t) 
            *t*=2;
        return task;
    }
    void  svc_end() {
        std::cout << "Stage2 completed: " << ff_node::get_my_id() << "\n";
    }
};

class Stage3: public ff_node {
public:
    int svc_init() {
        std::cout << "Stage3 starting\n";
        return 0;
    }
    
    void * svc(void * task) {
        unsigned int * t = (unsigned int *)task;
        
        if (t) {
            std::cout << "-> " << *t << "\n";
            free(t);
        }
        return task;
    }
    void  svc_end() {
        std::cout << "Stage3 completed: " << ff_node::get_my_id() << "\n";
    }
};

int main(int argc, char * argv[]) {
    if (argc!=2) {
        std::cerr << "use: "  << argv[0] << " streamlen\n";
        return -1;
    }
    
    // bild a 2-stage pipeline
    ff_pipeline pipe;
    pipe.add_stage(new Stage1(atoi(argv[1])));
    pipe.add_stage(new Stage2());
    pipe.add_stage(new Stage3());

    ffTime(START_TIME);
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    ffTime(STOP_TIME);

    std::cerr << "DONE, pipe  time= " << pipe.ffTime() << " (ms)\n";
    std::cerr << "DONE, total time= " << ffTime(GET_TIME) << " (ms)\n";
    pipe.ffStats(std::cerr);
    return 0;
}
