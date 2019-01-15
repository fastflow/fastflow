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
 * This test shows how to implement a complete user level scheduling policy.
 *
 */
#include <vector>
#include <iostream>
#include <algorithm> // for min_element
#include <ff/ff.hpp>
#include <ff/allocator.hpp>
#include <ff/utils.hpp>  

using namespace ff;

#define MAX_TIME_MS  999
#define N              3

// generic worker
class Worker: public ff_node {
public:
    void * svc(void * task) {
        int * t=(int*)task;

        tempo+=*t;
        //std::cerr << "WORKER received task tempo= " << tempo << "\n";
        ticks_wait(*t);
        return task;
    }

    void svc_end() {
        std::cerr << "WORKER time elapsed=" << tempo << " (ticks)\n";
    }
        
private:
    unsigned int tempo;
};


// emitter filter
class Emitter: public ff_monode {
public:
    Emitter(int maxtasks, int nworkers):
        maxtasks(maxtasks),ntask(0),nworkers(nworkers),load(nworkers,0) {}

    int svc_init() {
        srandom(::getpid()+(getusec()%4999));
        return 0;
    }

    void * svc(void * task) {
        if (task == NULL) {
            for(int i=0;i<nworkers;++i) {
                int * t = new int(random() % MAX_TIME_MS);
                load[i] += *t;
                ff_send_out_to(t,i);
                ++ntask;
            }
            return GO_ON;
        }
        
        if (ntask+N >= maxtasks) return NULL;

        for(int i=0;i<N;++i) {
            int * t = new int(random() % MAX_TIME_MS);

            /* this is my scheduling policy */
            std::vector<int>::iterator idx_it = std::min_element(load.begin(),load.end());
            long idx = static_cast<long>(idx_it - load.begin());
            load[idx] += *t;
            ff_send_out_to(t, idx);
            ++ntask;
        }

        return GO_ON;
    }
    
private:
    int maxtasks;
    int ntask;
    int nworkers;
    std::vector<int> load;
};



int main(int argc, char * argv[]) {
    int nworkers = 3;
    int ntask = 1000;
    if (argc>1) {
        if (argc<3) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nworkers ntask\n";
            return -1;
        }
        
        nworkers=atoi(argv[1]);
        ntask=atoi(argv[2]);
    }
    if (nworkers<=0 || ntask<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    ff_farm farm(false, 1024,8192);
    Emitter emitter(ntask,nworkers);
    farm.add_emitter(&emitter);

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);

    // set master_worker mode 
    farm.wrap_around();

    farm.run_and_wait_end();

    std::cerr << "DONE\n";
    return 0;
}
