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
 * NxN integer matrix multiplication.
 *
 */
#include <vector>
#include <iostream>
#include <farm.hpp>
#include <node.hpp>
#include <allocator.hpp>

  
using namespace ff;

const int N=1024;

static unsigned long A[N][N];
static unsigned long B[N][N];
static unsigned long C[N][N];

//#define USE_FFA
#if defined(USE_FFA)
static ff_allocator ffa;
#endif

struct task_t {
    task_t(int i,int j):i(i),j(j) {}
    int i;
    int j;
};


// generic worker
class Worker: public ff_node {
public:
#if defined(USE_FFA)
    int svc_init() {
        if (ffa.register4free()<0) {
            error("Worker, register4free fails\n");
            return -1;
        }
        return 0;
    }
#endif

    void * svc(void * task) {
        task_t * t = (task_t *)task;
        
        register unsigned int _C=0;
        for(register int k=0;k<N;++k)
            _C += A[t->i][k]*B[k][t->j];

        C[t->i][t->j] = _C;
#if defined(USE_FFA)        
        ffa.free(t);
#else
        delete t;
#endif
        return GO_ON;
    }
};

int main(int argc, 
         char * argv[]) {
    
    if (argc<2) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " nworkers\n";
        return -1;
    }
    
    int nworkers=atoi(argv[1]);
    
    if (nworkers<=0) {
        std::cerr << "Wrong parameter value\n";
        return -1;
    }
    
    /* init */
    for(int i=0;i<N;++i) 
        for(int j=0;j<N;++j) {
            A[i][j] = i+j;
            B[i][j] = i*j;
            C[i][j] = 0;
        }
        
    ffTime(START_TIME);

#if defined(USE_FFA)
	const struct { int nslab; int mincachedseg; } _nslabs[N_SLABBUFFER] = { 
	    { 16384, 8}, { 1, 0}, { 1, 0}, { 1, 0},{ 1, 0}, { 1, 0}, { 1, 0}, { 1, 0}, { 1, 0}}; 
	std::pair<int,int> nslabs[N_SLABBUFFER];                            
	for (int i=0;i<N_SLABBUFFER; ++i)                                   
	    nslabs[i]=std::make_pair(_nslabs[i].nslab, _nslabs[i].mincachedseg); 
    if (ffa.init(nslabs)<0) return -1;
    if (ffa.registerAllocator()<0) return -1;
#endif
    ff_farm<> farm(true /* accelerator set */, N*N+nworkers);    

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);    
    //farm.add_collector(new Collector);
    // Now run the accelator asynchronusly
    farm.run_then_freeze();
    for (int i=0;i<N;i++) {
        for(int j=0;j<N;++j) {
#if defined(USE_FFA)
            void * t = ffa.malloc(sizeof(task_t));
            task_t * task = new (t) task_t(i,j);
#else
            task_t * task = new task_t(i,j);
#endif
            farm.offload(task); 
        }
    }
    std::cout << "[Main] EOS arrived\n";
    farm.offload((void *)FF_EOS);
    
    // Here join
    farm.wait();  
    std::cout << "[Main] Farm accelerator stopped\n";
    
    ffTime(STOP_TIME);
    std::cerr << "DONE, farm time= " << farm.ffTime() << " (ms)\n";
    std::cerr << "DONE, total time= " << ffTime(GET_TIME) << " (ms)\n";
    farm.ffStats(std::cerr);
#if defined(USE_FFA)
    ffa.printstats();
#endif

#if 0
    for(int i=0;i<N;++i)  {
        for(int j=0;j<N;++j)
            printf(" %ld", C[i][j]);
        
        printf("\n");
    }
#endif


    return 0;
}


