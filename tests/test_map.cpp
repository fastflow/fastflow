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
 * The program is a simple map used:
 *   1. as a software accelerator (mapSA)
 *   2. to compute just one single task (mapOneShot).
 *
 */
#include <vector>
#include <iostream>
#include <ff/map.hpp>
#include <ff/utils.hpp>  

using namespace ff;

// this is the map function
void* mapF(basePartitioner*const P, int tid) {
    LinearPartitioner<int>* const partitioner=(LinearPartitioner<int>* const)P;
    LinearPartitioner<int>::partition_t Partition;
    partitioner->getPartition(tid, Partition);
    
    int* p   = (int*)Partition.getData(); // gets the pointer to the first element of the partion
    size_t l = Partition.getLength(); // gets the length of the partion

    for(size_t i=0;i<l;++i)  p[i] += tid;

    return p;  // returns the partition pointer !!!
}

long f(long v) { return v+1;}


int main(int argc, char * argv[]) {
    
    if (argc<3) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " arraysize nworkers\n";
        return -1;
    }
    int arraySize= atoi(argv[1]);
    int nworkers = atoi(argv[2]);

    if (nworkers<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    // defining a linear partitioner of integer elements
    LinearPartitioner<int> P(arraySize,nworkers);
    // defining the map passing as parameter the function called 
    // within each worker and the partitioner that has to be used
    // to create the worker partitions
    ff_map mapSA(mapF,&P, NULL, true); // the 4rd parameter enable the accelerator mode
    mapSA.run();

    printf("\nmapSA:\n");
    for(int i=0;i<10;++i) {
        // create a task
        int* A=new int[arraySize];
        for(int j=0;j<arraySize;++j) A[j]=i;

        // offload the task into the map 
        mapSA.offload(A);

        // wait for the result (NOTE: also the non-blocking call may be used !)
        void* R=NULL;
        mapSA.load_result(&R);

        // print the result
        for(int j=0;j<arraySize;++j) printf("%d ", ((int*)R)[j]);
        printf("\n");
    }
    // stopping the accelerator
    mapSA.offload(EOS);
    mapSA.wait();

    long *oneTask=new long[arraySize];
    for(long j=0;j<arraySize;++j) oneTask[j]=j;
#if __cplusplus > 199711L
    FF_MAP(map, oneTask,arraySize, f, nworkers);
#else
    ff_map mapOneShot(mapF,&P, oneTask);
    mapOneShot.run_and_wait_end();
#endif


    //ff_map mapOneShot(mapF,&P, oneTask);
    //mapOneShot.run_and_wait_end();
    // print the result
    printf("\nmapOneShot:\n");
    for(int j=0;j<arraySize;++j) printf("%ld ", oneTask[j]);
    printf("\n");

    std::cerr << "DONE\n";
    return 0;
}
