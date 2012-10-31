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
 * The program is a pipeline of 3 stages where the middle one is a simple map.
 * Logically:
 *
 *     node0 ---> map ---> node1
 *
 */
#include <vector>
#include <iostream>
#include <ff/map.hpp>
#include <ff/pipeline.hpp>
#include <ff/node.hpp>
#include <ff/utils.hpp>  

using namespace ff;

// this is the map function
void* mapF(basePartitioner*const P, int tid) {
    LinearPartitioner<int>* const partitioner=(LinearPartitioner<int>* const)P;
    LinearPartitioner<int>::partition_t Partition;
    partitioner->getPartition(tid, Partition);

    int* p = (int*)(Partition.getData());
    size_t l = Partition.getLength();

    for(size_t i=0;i<l;++i)  p[i] += tid;

    return p; // returns the partition pointer !!!
}


// first stage of the pipeline (it generates the stream of arrays)
class Init: public ff_node {
public:
    Init(int arraySize,int streamlen):arraySize(arraySize),streamlen(streamlen) {}
    
    void *svc(void *) {
        for(int i=0;i<streamlen;++i) {
            int * M = new int[arraySize];
            for(int j=0;j<arraySize;++j) M[j]=i;
            ff_send_out(M);
        }
        return NULL;
    }
private:
    int arraySize;
    int streamlen;
};

// last stage of the pipeline (it collects the stream of arrays)
class End: public ff_node {
public:
    End(int arraySize):arraySize(arraySize) {}
    void *svc(void *task) {
        int *M=(int*)task;
        for(int i=0;i<arraySize;++i)
            printf("%d ",M[i]);
        printf("\n");
        return GO_ON;
    }
private:
    int arraySize;
};

int main(int argc, char * argv[]) {
    
    if (argc<4) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " arraysize nworkers streamlen\n";
        return -1;
    }
    int arraySize=atoi(argv[1]);
    int nworkers =atoi(argv[2]);
    int streamlen=atoi(argv[3]);
    if (nworkers<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    
    // defining a linear partitioner of integer elements
    LinearPartitioner<int> P(arraySize,nworkers);
    // defining the map passing as parameter the function called 
    // within each worker and the partitioner that has to be used
    // to create the worker partitions
    ff_map map(mapF,&P);

    // defining the 3-stage pipeline
    ff_pipeline pipe;
    pipe.add_stage(new Init(arraySize,streamlen));
    pipe.add_stage(&map);
    pipe.add_stage(new End(arraySize));

    pipe.run_and_wait_end();

    std::cerr << "DONE\n";
    return 0;
}
