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

/**

 Very basic test for the FastFlow mapping utilities.
 
*/
#include <vector>
#include <iostream>
#include <ff/farm.hpp>

#include <sstream>
#define DEBUG
#ifdef DEBUG
#define DPRINT(s) \
{ \
    std::ostringstream ss; \
    ss << "[" << pthread_self() << "]: " << s << std::endl; \
    std::clog << ss.str(); \
}
#else
    #define DPRINT(s)
#endif


using namespace ff;

// generic worker
class Worker: public ff_node {
public:
    void * svc(void * task) {
        
        int * t = (int *)task;

        ff_mapThreadToCpu(2, 0);
        
        DPRINT( "Worker [" << ff_node::get_my_id() << "] : received task " << *t );

        DPRINT( "Worker ["<< ff_node::get_my_id() <<"] is mapped to the POSIX thread ID as : " << ff_getThreadID() );

        DPRINT( "Worker ["<< ff_node::get_my_id() <<"] is mapped to the core ID : " << ff_getMyCore() );

        return task;
    }
    // I don't need the following functions for this test
    //int   svc_init() { return 0; }
    //void  svc_end() {}

};

// the gatherer filter
class Collector: public ff_node {
public:
    void * svc(void * task) {
        int * t = (int *)task;
        if (*t == -1) return NULL;

        ff_mapThreadToCpu(3, 0);
        
        DPRINT( "Collector [" << ff_node::get_my_id() << "] is mapped to the POSIX thread ID as : " << ff_getThreadID() );
        
        DPRINT( "Collector [" << ff_node::get_my_id() << "] is mapped to the core ID : " << ff_getMyCore() );
        
        return task;
    }
};

// the load-balancer filter
class Emitter: public ff_node {
public:
    Emitter(int max_task):ntask(max_task) {};

    void * svc(void *) {
        int * task = new int(ntask);
        --ntask;
        if (ntask<0) return NULL;
        
        ff_mapThreadToCpu(1, 0);

        DPRINT( "Emitter [" << ff_node::get_my_id() << "] is mapped to the POSIX thread ID as : " << ff_getThreadID() );
        
        DPRINT( "Emitter [" << ff_node::get_my_id() <<"] is mapped to the core ID : " << ff_getMyCore() );
        
        return task;
    }
private:
    int ntask;
};


int main(int argc, char * argv[]) {

    if (argc<3) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " nworkers streamlen\n";
        return -1;
    }
    std::cout << "*********************************************" << "\n";
    
    std::cout << "The number of cores on the host machine are : "
              << ff_numCores() << "\n";

    std::cout << "The frequency of the core where the main function is running : "
              << ff_getCpuFreq() << "\n";

    std::cout << "The core ID where the main function is running : "
              << ff_getMyCore() << "\n";

    std::cout << "The cache line size of the core : " << cache_line_size() << "\n";

    std::cout << "*********************************************" << "\n";

    int nworkers=atoi(argv[1]);
    int streamlen=atoi(argv[2]);

    if (!nworkers || !streamlen) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    ff_farm<> farm; // farm object
    
    Emitter E(streamlen);
    farm.add_emitter(&E);

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w); // add all workers to the farm

    Collector C;
    farm.add_collector(&C);
    
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    std::cerr << "DONE, time= " << farm.ffTime() << " (ms)\n";
    farm.ffStats(std::cerr);

    return 0;
}
