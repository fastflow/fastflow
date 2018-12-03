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
 * Very basic test for the FastFlow pipeline in the accelerator configuration.
 *
 */
#include <vector>
#include <iostream>
#include <ff/ff.hpp>  
using namespace ff;

// generic worker
class Stage: public ff_node {
public:
    void * svc(void * task) {
        int * t = (int *)task;
        std::cout << "Stage " << ff_node::get_my_id() 
                  << " received task " << *t << "\n";
        return task;
    }
};


int main(int argc, char * argv[]) {
    int nstages = 3;
    int streamlen=1000;
    if (argc>1) {
        if (argc<3) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nstages streamlen\n";
            return -1;
        }
        
        nstages=atoi(argv[1]);
        streamlen=atoi(argv[2]);
    }
    if (nstages<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    ff_pipeline pipe(true /* accelerator flag */);
    for(int i=0;i<nstages;++i) {
        pipe.add_stage(new Stage);
    }

    ffTime(START_TIME);


    for(int j=0;j<3;++j) { // just 3 iterations to check if all is working 

        // Now run the accelator asynchronusly
        pipe.run_then_freeze(); //  pipe.run() can also be used here
        
        void * result=NULL;
        for (int i=0;i<streamlen;i++) {
            std::cout << "[Main] Offloading " << i << "\n"; 
            // Here offloading computation into the pipeline
            if (!pipe.offload(new int(i))) {
                error("offloading task\n");
                return -1;
            }
            
            // try to get results, if there are any
            if (pipe.load_result_nb(&result)) {
                std::cerr << "result= " << *((int*)result) << "\n";
                delete ((int*)result);
            }
        }

        std::cout << "[Main] send EOS\n";
        pipe.offload((void *)FF_EOS);
    
        // get all remaining results syncronously. 
        while(pipe.load_result(&result)) {
            std::cerr << "result= " << *((int*)result) << "\n";
            delete ((int*)result);
        }

        std::cout << "[Main] got all results iteration= " << j << "\n";
    
        // wait EOS
        if (pipe.wait_freezing()<0) {
            error("freezing error\n");
            return -1;
        }
    }

    // join all threads
    if (pipe.wait()<0) {
        error("error waiting pipe\n");
        return -1;
    }
    std::cout << "[Main] Pipe accelerator stopped\n";
    ffTime(STOP_TIME);
    std::cerr << "[Main] DONE, pipe time= " << pipe.ffTime() << " (ms)\n";
    std::cerr << "[Main] DONE, total time= " << ffTime(GET_TIME) << " (ms)\n";
    
    return 0;
}
