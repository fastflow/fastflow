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
 * Very basic test for the FastFlow pipeline (actually a 2-stage torus).
 *
 */

#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

struct ff_task_t {
    ff_task_t() {}
    int payload;
};

// generic stage
class Stage: public ff_node {
public:
    Stage(unsigned int streamlen):streamlen(streamlen),sum(0)  {}

    void * svc(void * task) {
        unsigned int * t = (unsigned int *)task;
        
        if (!t) {
            t = (unsigned int*)malloc(sizeof(int));
            if (!t) abort();
            
            *t=0;
            task = t;
        } else { sum+=*t; *t+=1;}

        if (*t == streamlen) return NULL;
        task = t;
        return task;
    }
    void  svc_end() {
        if (ff_node::get_my_id()) 
            std::cout << "Sum: " << sum << "\n";
    }

private:
    unsigned int streamlen;
    unsigned int sum;
};


int main() {
    int streamlen = 1000;
    // bild a 2-stage pipeline
    ff_pipeline pipe;
    pipe.add_stage(new Stage(streamlen));
    pipe.add_stage(new Stage(streamlen));

    pipe.wrap_around();
    pipe.cleanup_nodes();

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
