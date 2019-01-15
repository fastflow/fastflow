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
 * Very basic test for the FastFlow pipeline+farm.
 *
 */

#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

#include <vector>
using namespace std;


// generic stage
struct Stage: ff_node {
    void * svc(void * task) {
        return task;
    }
};

struct Emitter   : ff_node {
    Emitter(unsigned streamlen):streamlen(streamlen){}
    void *svc(void *in) {
        if (in== nullptr) {
            for(unsigned i=0;i<streamlen;++i)
                ff_send_out(new unsigned int(i));
            return EOS;
        }
        return GO_ON;
    }
    unsigned int streamlen;
};

struct Collector : ff_node {
    void * svc(void *task) {
        ff_send_out(task);
        return GO_ON;
    }
};



int main(int argc, char * argv[]) {
    int streamlen = 1000;
    if (argc>1) {
        if (argc!=2) {
            std::cerr << "use: "  << argv[0] << " streamlen\n";
            return -1;
        }
        streamlen = atoi(argv[1]);
    }
    
    // build a 2-stage pipeline
    ff_pipeline pipe;
    ff_farm f1, f2;
    vector<ff_node *> w1(1, new Stage);
    vector<ff_node *> w2(1, new Stage);
    f1.add_emitter(new Emitter(streamlen));
    f1.add_workers(w1);
    f1.add_collector(new Collector());
    f2.add_workers(w2);
    f2.add_collector(new Collector());
    pipe.add_stage(&f1);
    pipe.add_stage(&f2);

    if (pipe.wrap_around()<0) {
        error("wrap_around\n");
        return -1;
    }
    
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
