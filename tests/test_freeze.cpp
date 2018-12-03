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
 * Very basic test for the FastFlow farm (without the collector) in the 
 * accelerator configuration featuring freezing mode.
 *
 */
#include <vector>
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

// generic worker
class Worker: public ff_node {
public:
    Worker():ntask(0) {}

    int svc_init() {
        ntask=0;
        return 0;
    }

    void * svc(void * task) {
        ++ntask;
        return task;
    }

private:
    int ntask;
};

// the gatherer filter
class Collector: public ff_node {
public:
    Collector():ntask(0) {}

    int svc_init() {
        ntask=0;
        return 0;
    }
    void * svc(void * task) {        
        ++ntask;
        delete ((int*)task);
        return GO_ON;
    }
private:
    int ntask;
};

int main(int argc, char * argv[]) {
    int nworkers = 3;
    int streamlen= 1000;
    int iterations = 3;
    if (argc>1) {
        if (argc<4) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nworkers streamlen iteration\n";
            return -1;
        }        
        nworkers=atoi(argv[1]);
        streamlen=atoi(argv[2]);
        iterations=atoi(argv[3]);
    }
    if (nworkers<=0 || streamlen<=0 || iterations<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    ff_farm farm(true /* accelerator set */);
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);

    Collector C;
    farm.add_collector(&C);   


    for(int i=0;i<iterations;++i) {        
        // Now run the accelator asynchronusly 
        // and freeze it as soon as EOS is received
        farm.run_then_freeze();  
        std::cout << "[Main] Farm accelerator started\n";
    
        for (int i=0;i<streamlen;i++) {
            int * ii = new int(i);
            // Here offloading computation onto the farm
            farm.offload(ii); 
        }
        std::cout << "[Main] EOS arrived\n";
        void * eos = (void *)FF_EOS;
        farm.offload(eos);    
        // Here join
        farm.wait_freezing();          
        std::cout << "[Main] Farm accelerator frozen, time= " << farm.ffTime() << " (ms)\n";
    }
    
    std::cout << "[Main] Farm accelerator iterations completed\n";
    farm.wait();
    std::cout << "[Main] Farm accelerator stopped\n";

    std::cerr << "[Main] DONE\n";
    return 0;
}
