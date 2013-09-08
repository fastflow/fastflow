/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \link
 *  \file ff_matmul.cpp
 *  \ingroup application_level
 *
 *  \brief This file contains implementation of Matrix Multiplication program
 *  written in FastFlow
 *
 *  NxN integer matrix multiplication.
 *
 */

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

#define USE_FFA

#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/node.hpp>
#if defined(USE_FFA)
#include <ff/allocator.hpp>
#endif

/*!
 *  \ingroup application_level
 *
 *  @{
 */

using namespace ff;

const int N=512;

static unsigned long A[N][N];
static unsigned long B[N][N];
static unsigned long C[N][N];

#if defined(USE_FFA)
static ff_allocator ffa;
#endif

struct ff_task_t {
    ff_task_t(int i,int j):i(i),j(j) {}
    int i;
    int j;
};


/*!
 * \class Worker
 * \ingroup application_level
 *
 * \brief generic worker
 *
 * This class is defined in \ref ff_matmul.cpp
 *
 */

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
        ff_task_t * t = (ff_task_t *)task;
        register unsigned long _C=0;
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
    
    bool check=false;

    if (argc<2) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " nworkers\n";
        return -1;
    }
    
    if (argc==3) check=true;

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
    int nslabs[N_SLABBUFFER] = { N,0,0,0,0,0,0,0,0}; 
    if (ffa.init(nslabs)<0) return -1;
    if (ffa.registerAllocator()<0) return -1;
#endif
    ff_farm<> farm(true, N*N+nworkers);    

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker);
    farm.add_workers(w);    

    // Now run the accelator asynchronusly
    farm.run_then_freeze();
    for (int i=0;i<N;i++) {
        for(int j=0;j<N;++j) {
#if defined(USE_FFA)
            void * t = ffa.malloc(sizeof(ff_task_t));
            ff_task_t * task = new (t) ff_task_t(i,j);
#else
            ff_task_t * task = new ff_task_t(i,j);
#endif
            farm.offload(task); 
        }
    }
    std::cout << "[Main] EOS arrived\n";
    farm.offload((void *)FF_EOS);

#if defined(USE_FFA)
    ffa.deregisterAllocator();
#endif    
    
    // Here join
    farm.wait_freezing();  
    std::cout << "[Main] Farm accelerator stopped\n";
    
    ffTime(STOP_TIME);
    std::cerr << "DONE, farm time= " << farm.ffTime() << " (ms)\n";
    std::cerr << "DONE, total time= " << ffTime(GET_TIME) << " (ms)\n";
    farm.ffStats(std::cerr);
#if defined(USE_FFA) && defined(ALLOCATOR_STATS)
    ffa.printstats();
#endif

#if 0
    for(int i=0;i<N;++i)  {
        for(int j=0;j<N;++j)
            printf(" %ld", C[i][j]);
        
        printf("\n");
    }
#endif

    if (check) {
        unsigned long R=0;

        for(int i=0;i<N;++i) 
            for(int j=0;j<N;++j) {
                for(int k=0;k<N;++k)
                    R += A[i][k]*B[k][j];

                if (C[i][j]!=R) {
                    std::cerr << "Wrong result\n";
                    return -1;
                }
                R=0;
            }
        std::cout << "OK\n";
    }


    return 0;
}

/*!
 *  @}
 *  \endlink
 */
