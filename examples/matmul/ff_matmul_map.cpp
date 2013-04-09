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
#include <ff/partitioners.hpp>
#include <ff/map.hpp>
  
using namespace ff;

const unsigned long N=1024;


struct ff_task {
    ff_task(unsigned long* A, unsigned long* B, unsigned long* C):A(A),B(B),C(C) {}
    unsigned long* A;
    unsigned long* B;
    unsigned long* C;
};

class squareMatmulPartitioner: public basePartitioner {
public:
    typedef basePartitionList partition_t;

    squareMatmulPartitioner(size_t nElements, int nThreads):
         task(NULL), 
         p((nElements>(size_t)nThreads) ? nThreads : nElements), 
         q(nElements / nThreads), r(nElements % nThreads) {}

    inline void getPartition(const int threadId, basePartition& P) {
        const size_t start = (threadId * q) + ((r >= (size_t)threadId) ? threadId : r);
        basePartitionList & pList = *(basePartitionList*)&P;
        ff_task* t = (ff_task*)task;
        pList[0].setData(t->A+(start*N));
        pList[1].setData(t->B);
        pList[2].setData(t->C+(start*N));
        pList[0].setLength(((size_t)threadId<r)?(q+1):q);
    }

    inline void setTask(void* t) { task = (ff_task*)t; }
    inline size_t getParts() const { return p; }

protected:
    ff_task* task;
    const size_t p; 
    const size_t q; 
    const size_t r; 
};


// this is the map function
void* mapF(basePartitioner*const P, int tid) {
    squareMatmulPartitioner* const partitioner=(squareMatmulPartitioner* const)P;
    squareMatmulPartitioner::partition_t Partition;
    partitioner->getPartition(tid, Partition);
    
    unsigned long* A= (unsigned long*)Partition[0].getData();
    unsigned long* B= (unsigned long*)Partition[1].getData();
    unsigned long* C= (unsigned long*)Partition[2].getData();
    unsigned long  l= Partition[0].getLength();

    unsigned long _C=0;
    for(unsigned long i=0;i<l;++i) 
        for(unsigned long j=0;j<N;++j) {
            for(unsigned long k=0;k<N;++k)
#if defined(OPTIMIZE_CACHE)
                _C += A[j*N+i]*B[i*N+k];
            C[j*N+k]=_C;
#else
                _C += A[i*N+k]*B[k*N+j];
            C[i*N+j]=_C;
#endif

            _C=0;
        }
    return Partition.task;
}


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

    unsigned long* A = (unsigned long*)malloc(N*N*sizeof(unsigned long));
    unsigned long* B = (unsigned long*)malloc(N*N*sizeof(unsigned long));
    unsigned long* C = (unsigned long*)malloc(N*N*sizeof(unsigned long));
    assert(A && B && C);
    
    /* init */
    for(unsigned long i=0;i<N;++i) 
        for(unsigned long j=0;j<N;++j) {
            A[i*N+j] = i+j;
            B[i*N+j] = i*j;
            C[i*N+j] = 0;
        }
        
    ffTime(START_TIME);
    ff_task task(A,B,C);
    squareMatmulPartitioner P(N,nworkers);
    ff_map map(mapF, &P, &task);
    map.run_and_wait_end();
    ffTime(STOP_TIME);
    std::cerr << "DONE, map time= " << map.ffTime() << " (ms)\n";
    std::cerr << "DONE, total time= " << ffTime(GET_TIME) << " (ms)\n";

#if 0
    for(unsigned long i=0;i<N;++i)  {
        for(unsigned long j=0;j<N;++j)
            printf(" %ld", C[i*N+j]);
        
        printf("\n");
    }
#endif

    if (check) {
        unsigned long R=0;
        
        for(unsigned long i=0;i<N;++i) 
            for(unsigned long j=0;j<N;++j) {
                for(unsigned long k=0;k<N;++k)
                    R += A[i*N+k]*B[k*N+j];
                
                if (C[i*N+j]!=R) {
                    std::cerr << "Wrong result\n";
                    return -1;
                }
                R=0;
            }
        std::cout << "OK\n";
    }


    return 0;
}


