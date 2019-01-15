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
 * Tests nesting accelerators.
 *                                            --------------------------------------
 *                                           |                                      |
 *                           |Worker|        |               | Worker|              |
 *                           |               V               |                      |
 *    main-flow ---->Emitter |Worker|---Collector--->Emitter | Worker|--Collector---
 *        .                  |               |               |
 *        .                  |Worker|        |               | Worker|
 *        .                                  |
 *    main-flow <----------------------------          
 *  
 *          (main accelerator)                            (nested accelerator)         
 *
 */
#include <vector>
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

// generic worker
class Worker: public ff_node {
public:
    Worker(const char *tag): tag(tag) {}
    int svc_init() { std::cout << "[Worker" << tag << "] " << ff_node::get_my_id() << " started\n"; return 0; } 
    void * svc(void * task) {
        int * t = (int *)task;
        std::cout << "[Worker" << tag << "] " << ff_node::get_my_id() 
                  << " received task " << *t << "\n";
        return task;
    }
private:
    const char* tag;
};

// the gatherer filter
class Collector: public ff_node {
public:
    Collector(ff_farm * f):secondFarm(f) {}

    int svc_init() {
        if (secondFarm==NULL) return 0;
        else {
            std::cerr << "Starting 2nd farm\n";
            secondFarm->run_then_freeze();
        }
        return 0;
    }

    void * svc(void * task) {   
        void * result=NULL;
        int * t = (int *)task;
        
        if (secondFarm) {
            std::cout << "[Farm Collector1] task received " << *t << "\n";
            secondFarm->offload(task);
            if (secondFarm->load_result_nb(&result)) {
                std::cerr << "result_nb = " << *((int*)result) << "\n";
                delete ((int*)result);
                return GO_ON;
            }
        } else {
            std::cout << "[Farm Collector2] task received " << *t << "\n";
        }
        return task;
    }

    void svc_end() {
        if (secondFarm)  {
            std::cerr << "SENDING EOS and WAITING RESULTS\n";
            secondFarm->offload((void *)FF_EOS);

            void * result;
            while(secondFarm->load_result(&result)) {
                std::cerr << "result = " << *((int*)result) << "\n";
                delete ((int*)result);
            }
            secondFarm->wait();
        }
    }


private:
    ff_farm *secondFarm;
};

int main(int argc, char * argv[]) {
    int nworkers=4;
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

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker("1"));
    farm.add_workers(w);

    ff_farm farm2(true);
    std::vector<ff_node *> w2;
    for(int i=0;i<nworkers;++i) w2.push_back(new Worker("2"));
    farm2.add_workers(w2);
    Collector C2(NULL);
    farm2.add_collector(&C2);
    
    Collector C(&farm2);
    farm.add_collector(&C);   

    // Now run the accelator asynchronusly
    farm.run();
    std::cout << "[Main] Farm accelerator started\n";
    
    for (int i=0;i<streamlen;i++) {
        int * ii = new int(i);
        std::cout << "[Main] Offloading " << i << "\n"; 
        // Here offloading computation onto the farm
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
