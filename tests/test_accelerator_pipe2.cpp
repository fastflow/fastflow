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

struct Worker: ff_node_t<long> {
    long * svc(long *in) {
        std::cout << "Worker " << ff_node::get_my_id() 
                  << " received " << (long)in << "\n";
        return in;
    }
};


int main() {
    std::vector<ff_node*> W;
    W.push_back(new Worker);
    W.push_back(new Worker);
    
    ff_farm farm(W, nullptr, nullptr);
    farm.cleanup_all();
    ff_pipeline pipe0;
    pipe0.add_stage(farm);

    
    ff_pipeline pipe(true /* accelerator flag */);
    pipe.add_stage(pipe0);

    pipe.run();
    void * result=NULL;
    for (long i=1;i<100;++i) {
        if (!pipe.offload((long*)(i))) {
            error("offloading task\n");
            return -1;
        }
        
        // try to get results, if there are any
        if (pipe.load_result_nb(&result)) {
            std::cerr << "result= " << (long)result << "\n";
        }
    }
    pipe.offload(FF_EOS);
    // get all remaining results syncronously. 
    while(pipe.load_result(&result)) {
        std::cerr << "result= " << (long)result << "\n";
    }
    
    if (pipe.wait()<0) {
        error("waiting pipe termination\n");
        return -1;
    }
    return 0;
}
