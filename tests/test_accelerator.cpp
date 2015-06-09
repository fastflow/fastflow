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
 * Very basic test for the FastFlow farm in the accelerator configuration.
 *
 */
#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/node.hpp>
  
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
};


int main(int argc, char * argv[]) {
    int nworkers = 3;
    int streamlen= 1000;
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
    
    ff_farm<> farm(true /* accelerator set */);
    
    // Using standard emitter
    /*
      Emitter E(streamlen);
      farm.add_emitter(&E);
    */

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);

    farm.add_collector(NULL);

    ffTime(START_TIME);
    // Now run the accelator asynchronusly
    farm.run_then_freeze(); //  farm.run() can also be used here

    void * result=NULL;
    for (int i=0;i<streamlen;i++) {
        int * ii = new int(i);
        std::cout << "[Main] Offloading " << i << "\n"; 
        // Here offloading computation onto the farm
        farm.offload(ii); 

        // try to get results, if there are any
        if (farm.load_result_nb(&result)) {
            std::cerr << "result= " << *((int*)result) << "\n";
            delete ((int*)result);
        }
    }
    std::cout << "[Main] EOS arrived\n";
    farm.offload((void *)FF_EOS);
    
#if 0
    // get all remaining results syncronously. 
    while(farm.load_result(&result)) {
        std::cerr << "result= " << *((int*)result) << "\n";
        delete ((int*)result);
    }
#else
    // asynchronously wait results
    do {
        if (farm.load_result_nb(&result)) {
            if (result==(void*)FF_EOS) break;
            std::cerr << "result= " << *((int*)result) << "\n";
            delete ((int*)result);
        } 
    } while(1);
#endif
    
    // Here join
    farm.wait();  
    std::cout << "[Main] Farm accelerator stopped\n";

    ffTime(STOP_TIME);
    std::cerr << "[Main] DONE, farm time= " << farm.ffTime() << " (ms)\n";
    std::cerr << "[Main] DONE, total time= " << ffTime(GET_TIME) << " (ms)\n";
    
    return 0;
}
