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
/* Author: Massimo Torquati
 *         torquati@di.unipi.it  massimotor@gmail.com
 *
 *  farm(map);
 *
 * NOTE: the farm is not an ordered farm (in case use ff_ofarm).
 *
 */

#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif

#include <ff/stencilReduceOCL.hpp>
#include <ff/farm.hpp>

#define CHECK 1
#ifdef CHECK
#include "ctest.h"
#else
#define NACC 1
#endif

using namespace ff;

FF_OCL_MAP_ELEMFUNC(mapf, float, elem, useless,
                    (void)useless;
                    return (elem+1.0);
                    );

//implicit input
//FF_OCL_MAP_ELEMFUNC_1D(mapf, float, elem, return (elem+1.0));

// stream task
struct myTask {
    myTask(float *M, const size_t size):M(M),size(size)
#ifdef CHECK
    , expected_sum(0)
#endif
    {}
    float *M;
    const size_t size;
#ifdef CHECK
    float expected_sum;
#endif
};

// OpenCL task
struct oclTask: baseOCLTask<myTask, float> {
    oclTask() {}
    void setTask(myTask *task) { 
        assert(task);
        setInPtr(task->M, task->size);
        setOutPtr(task->M, task->size);
    }
};

struct Emitter: ff_node {
    Emitter(long streamlen, size_t size):streamlen(streamlen),size(size) {}
    void *svc(void*) {
        for(int i=0;i<streamlen;++i) {
            float* task = new float[size];            
            for(size_t j=0;j<size;++j) task[j]=j+i;
            myTask *T = new myTask(task, size);
#ifdef CHECK
            T->expected_sum = 0;
            for(size_t j=0; j<size; ++j)
            	T->expected_sum += task[j] + 1;
#endif
            ff_send_out(T);
        }
        return EOS;
    }
    long   streamlen;
    size_t size;
};

struct Collector: ff_node_t<myTask> {
    myTask* svc(myTask *t) {
#if defined(CHECK)
//        for(size_t i=0;i<t->size;++i)  printf("%.2f ", t->M[i]);
//        printf("\n");
    	float sum = 0;
    	for(size_t i=0; i<t->size; ++i)
    		sum += t->M[i];
    	check &= (sum == t->expected_sum);
    	//printf("sum = %f, exp = %f\n", sum, t->expected_sum);
#endif
        delete [] t->M; delete t;
        return GO_ON;
    }
#ifdef CHECK
    bool check;
    Collector() : check(true) {}
#endif
};

struct Worker: ff_mapOCL_1D<myTask, oclTask> {
	Worker(std::string mapf, const size_t nacc) :
			ff_mapOCL_1D<myTask, oclTask>(mapf, nullptr, nacc) {
		SET_DEVICE_TYPE((*this));
	}
};


int main(int argc, char * argv[]) {
    
    size_t inputsize   = 1024;
    long   streamlen   = 2048;
    int    nworkers    = 3;
    
    if  (argc > 1) {
        if (argc < 4) { 
            printf("use %s arraysize streamlen nworkers\n", argv[0]);
            return 0;
        } else {
            inputsize = atol(argv[1]);
            streamlen = atol(argv[2]);
            nworkers = atoi(argv[3]);
        }
    }
    
    std::vector<std::unique_ptr<ff_node> > W;
    for(int i=0;i<nworkers;++i)  W.push_back(make_unique<Worker>(mapf,NACC));
    Emitter   E(streamlen,inputsize);
    Collector C;
    ff_Farm<> farm(std::move(W), E, C);
    farm.run_and_wait_end();
#if defined(CHECK)
    if (!C.check) {
    	printf("Wrong result\n");
    	exit(1); //ctest
    }
    else printf("OK\n");
#endif

    printf("DONE\n");
    return 0;
}
