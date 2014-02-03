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
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
using namespace ff;

#include <vector>
using namespace std;


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


class Emitter   : public ff_node {
public:
    Emitter(unsigned streamlen):streamlen(streamlen){}
    void *svc(void*) {
        for(unsigned i=0;i<streamlen;++i)
            ff_send_out(new unsigned int(i));
        return NULL;
    }
private:
    unsigned int streamlen;
};

class Collector : public ff_node {
public:
    void * svc(void *task) {
        return task;
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
    ff_farm<> f1, f2;
    vector<ff_node *> w1(1, new Stage(streamlen));
    vector<ff_node *> w2(1, new Stage(streamlen));
    f1.add_emitter(new Emitter(streamlen));
    f1.add_workers(w1);
    f1.add_collector(new Collector());
    f2.add_workers(w2);
    f2.add_collector(new Collector());
    pipe.add_stage(&f1);
    pipe.add_stage(&f2);

    pipe.wrap_around();

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
