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
 * Master-worker in accelerator configuration.
 * TODO: getting back results from the accelerator by using get_results.
 *
 *
 *                 ------------------------                  
 *   main         |              -----     |
 *    |           |        -----|     |    |
 *    |           v       |     |  W  |----
 *    |         -----     |      -----
 *    |------> |  E  |----|        .
 *    |        |     |    |        .
 *    |         -----     |        .
 *    |           ^       |      -----
 *    |           |        -----|  W  | ---
 *    -           |             |     |    |
 *                |              -----     |
 *                 ------------------------    
 *
 *
 */      
#include <vector>
#include <iostream>
#include <ff/ff.hpp>
  

using namespace ff;

struct W: ff_node {
    void *svc(void *task){
        std::cout << "W(" << get_my_id() << ") got task " << (*(ssize_t*) task) << "\n";
        return task;
    }
};

class E: public ff_node {
public:
    E(ff_loadbalancer *const lb):lb(lb) {}
    int svc_init() {
        eosreceived=false, numtasks=0;
        return 0;
    }
    void *svc(void *task) {	
        if (lb->get_channel_id() == -1) {
            ++numtasks;
            return task;
        }        
        if (--numtasks == 0 && eosreceived) return FF_EOS;
        return GO_ON;	
    }
    void eosnotify(ssize_t id) {
        if (id == -1)  {
            eosreceived = true;
            if (numtasks == 0) {
                printf("BROADCAST\n");
                fflush(stdout);
                lb->broadcast_task(FF_EOS);
            }
        }
    }
private:
    bool eosreceived;
    long numtasks;
protected:
    ff_loadbalancer *const lb;
};



int main(int argc,  char * argv[]) {
    int nworkers=3;
    int streamlen=1000;
    int iterations=3;
    if (argc>1) {
        if (argc<4) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nworkers streamlen iterations\n";
            return -1;
        }        
        nworkers=atoi(argv[1]);
        streamlen=atoi(argv[2]);
        iterations=atoi(argv[3]);
    }
    if (nworkers<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    ff_farm farm(true /* accelerator set */);
    E emitter(farm.getlb());
    farm.add_emitter(&emitter);

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new W);
    farm.add_workers(w);

    // set master_worker mode 
    farm.wrap_around();

    for(int i=0;i<iterations;++i) {        
        // Now run the accelator asynchronusly 
        // and freeze it as soon as EOS is received
        farm.run_then_freeze();  
        std::cout << "[Main] Farm accelerator started\n";
    
        for (ssize_t j=0;j<streamlen;j++) {
            ssize_t * ii = new ssize_t(j);
            // Here offloading computation onto the farm
			std::cout << "[Main] Offloading " << *ii << "\n";
            farm.offload(ii); 
        }
        farm.offload(FF_EOS);    
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
