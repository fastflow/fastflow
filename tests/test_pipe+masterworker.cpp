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
 *       3-stages pipeline
 *               
 *                                      -----------------------
 *                                     |                       |
 *             --> Worker1 -->         |        --> Worker2 ---
 *            |               |        v       |
 * Start ----> --> Worker1 -->  --> Emitter -- 
 *            |               |        ^       |
 *             --> Worker1 -->         |        --> Worker2 ---
 *                                     |                       |
 *                                      -----------------------
 */



#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>
#include <ff/node.hpp>
  

using namespace ff;

class Worker1: public ff_node {
public:
    void * svc(void * task) {
        usleep(10000);
        return task;
    }
};

class Worker2: public ff_node {
public:
    void * svc(void * task) {
        usleep(500);
        return task;
    }
};

class Emitter: public ff_node {
public:
    void * svc(void * task) {
        int * t = (int *)task;
        if (*t % 2)  return GO_ON;
        ++(*t);
        return task;
    }
};

class Start: public ff_node {
public:
    Start(int streamlen):streamlen(streamlen) {}
    void* svc(void*) {    
        for (int j=0;j<streamlen;j++) {
            int * ii = new int(j);
            ff_send_out(ii);            
        }
        return NULL;
    }
private:
    int streamlen;
};


int main(int argc, 
         char * argv[]) {

    if (argc<4) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " nworkers1 nworkers2 streamlen\n";
        return -1;
    }
    
    int nworkers1=atoi(argv[1]);
    int nworkers2=atoi(argv[2]);
    int streamlen=atoi(argv[3]);

    if (nworkers1<=0 || nworkers2<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    ff_pipeline pipe;
    
    Start start(streamlen);
    pipe.add_stage(&start);

    ff_farm<> farm1;
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers1;++i) w.push_back(new Worker1);
    farm1.add_workers(w);
    farm1.add_collector(NULL);

    pipe.add_stage(&farm1);


    ff_farm<> farm2;
    Emitter emitter;
    farm2.add_emitter(&emitter);

    w.clear();
    for(int i=0;i<nworkers2;++i) w.push_back(new Worker2);
    farm2.add_workers(w);

    // set master_worker mode 
    farm2.wrap_around();

    pipe.add_stage(&farm2);
    pipe.run_and_wait_end();

    std::cerr << "DONE\n";
    pipe.ffStats(std::cerr);
    return 0;
}
