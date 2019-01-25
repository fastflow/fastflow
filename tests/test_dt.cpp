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
#include <ff/ff.hpp>
#include <ff/allocator.hpp>  

using namespace ff;

static ff_allocator ffalloc;

// generic worker
class Worker: public ff_node {
public:
    void * svc(void * task) {
        return task;
    }
};

class Emitter: public ff_node {
public:
    Emitter(int maxtasks):maxtasks(maxtasks),ntask(0),first_k(4) {}

    int svc_init() {
        if (ffalloc.registerAllocator()<0) {
            error("Emitter, registerAllocator fails\n");
            return -1;
        }
        return 0;
    }

    void * svc(void * task) {
        if (task == NULL) {
            for(int i=0;i<first_k;++i) {
                int * t=(int*)ffalloc.malloc(sizeof(int));
                *t=i;
                ff_send_out(t);
                ntask++;
                std::cerr << "send_out task " << *t << " ntask=" << ntask << "\n";
            }
            return GO_ON;
        }
        std::cerr << "EMITTER received task= " << *(int*)task << "\n";
        --ntask;

        int * t = (int *)task;
        int   r=(*t %4);

        if ((*t+r)<=maxtasks) {
            if (r) {
                for(int i=1;i<=(r-1);++i) {
                    int * t2=(int*)ffalloc.malloc(sizeof(int));
                    *t2=*t+i;
                    ff_send_out(t2);
                    ntask++;
                    std::cerr << "send_out task " << *t2 << " ntask(2)=" << ntask << "\n";
                }
                
                (*t)+=r; ntask++;
                std::cerr << "send_out task " << *t << " ntask(3)=" << ntask << "\n";
                return task;            
            }
        }

        ffalloc.free(task);
        std::cerr << "task " << *(int*)task << " removed ntask=" << ntask << "\n";
        if (ntask==0) return NULL;

        return GO_ON;
    }
    
private:
    int maxtasks;
    int ntask;
    int first_k;
};



int main(int argc,  char * argv[]) {
    int nworkers=3;
    int ntask   =40;
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
    
    ffalloc.init();

    ff_farm farm(false,8192,8192);
    farm.set_scheduling_ondemand(); // set on-demand scheduling policy

    Emitter emitter(ntask);
    farm.add_emitter(&emitter);

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);

    // set master_worker mode 
    farm.wrap_around();

    if (farm.run_and_wait_end()<0)
        abort();

    std::cerr << "DONE\n";
    return 0;
}
