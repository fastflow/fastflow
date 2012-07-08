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
 * This test shows how to implement a simple map using a farm skeleton.
 * The program is a pipeline of 3 stages where the middle one is a map.
 * Logically:
 *
 *     node0 ---> map ---> node1
 *
 */
#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>
#include <ff/node.hpp>
#include <ff/allocator.hpp>
#include <ff/utils.hpp>  

using namespace ff;

const int SIZE = 64;


// *************** simple 1D array partitioner definition ****************** 

// contiguous range of indexes
class Range {
private:
    std::pair<int,int> range;
protected:
    std::pair<int,int>& get() { return range;}
public:
    Range()                { reset(); }
    Range(int b, int e)    { set(b,e); }
    Range(const Range & r) { set(r); }

    Range& operator=(const Range & r) { 
        range.first=r.begin(); range.second=r.end();
        return *this;
    }
    
    int begin() const { return range.first;}
    int end()   const { return range.second;}
    int size()  const { return range.second - range.first;}
    
    void set(int b, int e) { 
        assert(b>=0); assert(e>=0);
        range.first=b; range.second=e;
    }
    void set(const Range & r) { set(r.begin(),r.end()); }
    void reset() { range.first=-1; range.second=-1;}
    bool empty() const { 
        if (range.second == -1) return true;
        if (range.second < range.first) return true;
        return false;
    }
};

// one dimensional array partitioner
class Partitioner {
public:
    Partitioner(int nCols, int nThreads):
        nCols(nCols),nThreads(nThreads) {
    }
    
    inline void getPartition(const int threadId, Range & range) {
        int r = nCols / nThreads;
        const int m = nCols % nThreads;
        const int start = (threadId * r) + ((m >= threadId) ? threadId : m);
        if (threadId < m) ++r;
        range.set(start, (start+r)-1);
    }
    
    inline void set(int c, int t) {
        nCols = c, nThreads = t;
    }
    
protected:
    int  nCols, nThreads;
};
// *************************************************************************


// map's worker
class MapWorker: public ff_node {
public:
    MapWorker(Partitioner *const P):P(P) {}
    void * svc(void * task) {
        int * M=(int*)task;

        Range r;
        P->getPartition(ff_node::get_my_id(),r);

        printf("Worker:%d [%d..%d]\n", ff_node::get_my_id(),r.begin(),r.end());

        for(int i=r.begin();i<=r.end();++i) ++M[i];

        return task;
    }
private:
    Partitioner * const P;
};


// first stage of the pipeline (it generates the stream of arrays)
class Init: public ff_node {
public:
    Init(int ntasks):ntasks(ntasks) {}
    
    void *svc(void *) {
        for(int i=0;i<ntasks;++i) {
            int * M = new int[SIZE];
            for(int j=0;j<SIZE;++j) M[j]=i;
            ff_send_out(M);
        }
        return NULL;
    }
private:
    int ntasks;
};

// last stage of the pipeline (it collects the stream of arrays)
class End: public ff_node {
public:
    void *svc(void *task) {
        int *M=(int*)task;
        for(int i=0;i<SIZE;++i)
            printf("%d ",M[i]);
        printf("\n");
        return GO_ON;
    }
};

/* 
 * Needed to be able to call broadcast_task 
 */
class my_loadbalancer: public ff_loadbalancer {
public:
    // this is necessary because ff_loadbalancer has non default parameters....
    my_loadbalancer(int max_num_workers):ff_loadbalancer(max_num_workers) {}

    void broadcast(void * task) {
        ff_loadbalancer::broadcast_task(task);
    }   
};
/* 
 * Needed to be able to call all_gather 
 */
class my_gatherer: public ff_gatherer {
public:
    my_gatherer(int max_num_workers):ff_gatherer(max_num_workers) {}

    int all_gather(void * task, void **V) {
        return ff_gatherer::all_gather(task,V);
    }   
};

// emitter filter of the map, it just broadcasts data to the workers
class Emitter: public ff_node {
public:
    Emitter(my_loadbalancer * const lb): lb(lb) {}

    void * svc(void * task) {
        lb->broadcast(task);
        return GO_ON;
    }
private:
    my_loadbalancer * lb;
};

// collector filter of the map, it just collects data from each worker using
// the all_gather
class Collector: public ff_node {
public:
    Collector(my_gatherer * const gt): gt(gt) {}

    void * svc(void *task) {  
        void *Task[gt->getnworkers()];
        gt->all_gather(task, &Task[0]);
        return Task[0];
    }
private:
    my_gatherer *const gt;
};


int main(int argc, char * argv[]) {
    
    if (argc<3) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " nworkers ntasks\n";
        return -1;
    }
    
    int nworkers=atoi(argv[1]);
    int ntasks=atoi(argv[2]);
    if (nworkers<=0 || ntasks<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    Partitioner P(SIZE,nworkers);

    ff_farm<my_loadbalancer, my_gatherer> farm;

    Emitter emitter(farm.getlb());
    Collector collector(farm.getgt());
    farm.add_emitter(&emitter);
    farm.add_collector(&collector);

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new MapWorker(&P));
    farm.add_workers(w);

    ff_pipeline pipe;
    pipe.add_stage(new Init(ntasks));
    pipe.add_stage(&farm);
    pipe.add_stage(new End);

    pipe.run_and_wait_end();

    std::cerr << "DONE\n";
    return 0;
}
