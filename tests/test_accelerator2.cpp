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
 * accelerator configuration.
 *
 */
#include <vector>
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

// generic worker
class Worker: public ff_node {
public:
    void * svc(void * task) {
        int * t = (int *)task;
        std::cout << "[Worker] " << ff_node::get_my_id() 
                  << " received task " << *t << "\n";
        return task;
    }
    // I don't need the following for this test
    //int   svc_init() { return 0; }
    //void  svc_end() {}

};

// the gatherer filter
class Collector: public ff_node {
public:
    void * svc(void * task) {        
        int * t = (int *)task;
        std::cout << "[Farm Collector] task received " << *t << "\n";
        delete ((int*)task);
        return GO_ON;
    }
};

// the load-balancer filter 
/* 
class Emitter: public ff_node {
public:
    Emitter(int max_task):ntask(max_task) {
        std::cout << "Emitter set up\n";  
    };
    void * svc(void *intask) {     
        //std::cout << " EMITTER \n";
        return intask;   
    }
private:
    int ntask;
};
*/

int main(int argc, char * argv[]) {
    int nworkers = 3;
    int streamlen=1000;
    if (argc>1) {
        if (argc<3) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nworkers streamlen\n";
            return -1;
        }    
        nworkers=atoi(argv[1]);
        streamlen=atoi(argv[2]);
    }
    if (nworkers<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    ff_farm farm(true /* accelerator set */);
    std::vector<ff_node *> W;
    for(int i=0;i<nworkers;++i) W.push_back(new Worker);
    farm.add_workers(W);

    Collector C;
    farm.add_collector(&C);   
    

    // Now run the accelator asynchronusly
    farm.run();
    std::cout << "[Main] Farm accelerator started\n";
    
    for (int i=0;i<streamlen;i++) {
        int * ii = new int(i);
        std::cout << "[Main] Offloading " << i << "\n"; 
        // Here offloading computation into the farm
        farm.offload(ii); 
    }
    std::cout << "[Main] EOS arrived\n";
    void * eos = (void *)FF_EOS;
    farm.offload(eos);
    
    // Here join
    farm.wait();  

    std::cout << "[Main] Farm accelerator stopped\n";

    std::cerr << "[Main] DONE, time= " << farm.ffTime() << " (ms)\n";
    return 0;
}
