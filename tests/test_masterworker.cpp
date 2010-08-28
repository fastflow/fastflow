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
 * Very basic test for the FastFlow farm in the master-worker configuration.
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
        std::cerr << "sleeping....\n";
        usleep(10000);
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



int main(int argc, 
         char * argv[]) {

    if (argc<4) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " nworkers streamlen iterations\n";
        return -1;
    }
    
    int nworkers=atoi(argv[1]);
    int streamlen=atoi(argv[2]);
    int iterations=atoi(argv[3]);

    if (nworkers<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    ff_farm<> farm(true /* accelerator set */);
    Emitter emitter;
    farm.add_emitter(&emitter);

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);

    // set master_worker mode 
    farm.wrap_around();

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
    farm.ffStats(std::cerr);
    return 0;
}
