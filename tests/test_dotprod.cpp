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
 * The program computes the dot-product of 2 arrays.
 *
 */
#include <vector>
#include <iostream>
#include <ff/map.hpp>
#include <ff/partitioners.hpp>
#include <ff/utils.hpp>  

using namespace ff;

// task definition
struct ff_task_t {
    ff_task_t(unsigned long*A,unsigned long*B):A(A),B(B) {}
    unsigned long* A;
    unsigned long* B;
};

/* ------------------- definition of the partitioner for the problem ---------------------- */
struct myPartition: public basePartition {
    myPartition():task2(NULL) {}
    
    inline void   setFirstData(void* t) {task=t;}
    inline void   setSecondData(void* t) {task2=t;}
    inline void   setLength(size_t l) {len=l;}
    inline void*  getFirstData() { return task;}
    inline void*  getSecondData() { return task2;}
    inline size_t getLength() { return len;}
    
    void* task2;
};

template<typename T1, typename T2>
class myPartitioner: public LinearPartitioner<T1> {
    typedef LinearPartitioner<T1> base;
public:
    typedef myPartition partition_t;

    myPartitioner(size_t nElements, int nThreads): base(nElements,nThreads) {}
    inline void getPartition(const int threadId, basePartition& P) {
        const size_t start = 
            (threadId * base::q) + ((base::r >= (size_t)threadId) ? threadId : base::r);
        P.setLength(((size_t)threadId<base::r)?(base::q+1):base::q);
        myPartition* p = (myPartition*)&P;
        p->setFirstData(base::task+start);
        p->setSecondData(task2+start);
    }
    inline void setTask(void* t)  { 
        ff_task_t* mytask = (ff_task_t*)t;
        base::task = mytask->A;
        task2= mytask->B;
    }
protected:
    T2* task2;
};
/* ---------------------------------------------------------------------------------------- */


// this is the map function which computes the local reduce of the 2 partitions
void* mapF(basePartitioner*const P, int tid) {
    myPartitioner<int,int>* const partitioner=(myPartitioner<int,int>* const)P;
    myPartitioner<int,int>::partition_t Partition;
    partitioner->getPartition(tid, Partition);

    unsigned long* A = (unsigned long*)(Partition.getFirstData());
    unsigned long* B = (unsigned long*)(Partition.getSecondData());
    size_t l = Partition.getLength();

    unsigned long sum=0;
    for(size_t i=0;i<l;++i)  sum+= A[i]*B[i];

    return (new unsigned long(sum));
}

static unsigned long sum=0;

// this is the function called in the Collector for the final reduce
// V contains all the partial sums sent by the n workers
void* reduceF(void** V, int n) {
    for(int i=0;i<n;++i)  {
        sum += ((unsigned long**)V)[i][0];
        //delete (unsigned long*)V[i];
    }
    return NULL; // we don't want to send out any tasks
}



int main(int argc, char * argv[]) {    
    if (argc<3) {
        std::cerr << "use: " << argv[0] << " arraysize nworkers\n";
        return -1;
    }
    int arraySize= atoi(argv[1]);
    int nworkers = atoi(argv[2]);

    if (nworkers<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    // creates the array
    unsigned long* A = new unsigned long[arraySize];
    unsigned long* B = new unsigned long[arraySize];
    for(int j=0;j<arraySize;++j) { A[j]=j; B[j]=2*j;}

    // create the task
    ff_task_t* oneTask=new ff_task_t(A,B);
    // partitioner and map setup
    myPartitioner<unsigned long,unsigned long> P(arraySize,nworkers);
    ff_map mapreduce(mapF,&P,oneTask,reduceF);
#if 0
    mapreduce.setAffinity(0);
    // the worker_mapping array, should be dinamically built                                                                                                          
    const char worker_mapping[] = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31";
    threadMapper::instance()->setMappingList(worker_mapping);
#endif
    mapreduce.run_and_wait_end();
    printf("Sum= %lu\n", sum);
    printf("Time= %.4f (ms)\n", mapreduce.ffwTime()); 
    std::cerr << "DONE\n";
    return 0;
}
