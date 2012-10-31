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
 * The program is a mapreduce which computes the sum of the elements of the
 * array given in input.
 *
 */
#include <vector>
#include <iostream>
#include <ff/map.hpp>
#include <ff/utils.hpp>  

using namespace ff;

// this is the map function which computes a local reduce
void* mapF(basePartitioner*const P, int tid) {
    LinearPartitioner<int>* const partitioner=(LinearPartitioner<int>* const)P;
    LinearPartitioner<int>::partition_t Partition;
    partitioner->getPartition(tid, Partition);

    int* p = (int*)(Partition.getData());
    size_t l = Partition.getLength();

    for(size_t i=1;i<l;++i)  p[0]+=p[i]; 

    return &p[0];
}

// this is the reduce function called in the Collector
void* reduceF(void** V, int n) {
    int sum=0;
    for(int i=0;i<n;++i)  sum += ((int**)V)[i][0];
    printf("Sum = %d\n", sum);

    return NULL;
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
    int* oneTask=new int[arraySize];
    for(int j=0;j<arraySize;++j) oneTask[j]=j;

    LinearPartitioner<int> P(arraySize,nworkers);
    ff_map mapreduce(mapF,&P,oneTask,reduceF);
    mapreduce.run_and_wait_end();

    std::cerr << "DONE\n";
    return 0;
}
