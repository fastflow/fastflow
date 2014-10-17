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
 *  farm(reduce);
 *
 * NOTE: the farm is not an ordered farm (in case use ff_ofarm).
 */

#include <ff/farm.hpp>
#include <ff/mapOCL.hpp>

using namespace ff;

FFREDUCEFUNC(reducef, float, x,y, return (x+y) );

// stream task
struct myTask {
    myTask(float *M, const size_t size):M(M),sum(0.0),size(size) {}
    float *M;
    float  sum;
    const size_t size;
};

// OpenCL task
struct oclTask: public baseOCLTask<myTask, float> {
    oclTask() {}
    void setTask(const myTask *task) { 
        assert(task);
        setInPtr(task->M);
        setSizeIn(task->size);
        setReduceVar(&task->sum);
    }
};

struct Emitter: public ff_node {
    Emitter(long streamlen, size_t size):streamlen(streamlen),size(size) {}
    void* svc(void*) {
        for(int i=0;i<streamlen;++i) {
            float* task = new float[size];            
            for(size_t j=0;j<size;++j) task[j]=j+i;
            myTask *T = new myTask(task, size);
            ff_send_out(T);
        }
        return EOS;
    }
    long   streamlen;
    size_t size;
};

struct Collector: public ff_node_t<myTask> {
    myTask *svc(myTask *t) {
        printf("%.2f\n", t->sum);
        return GO_ON;
    }
};


int main(int argc, char * argv[]) {
    if (argc<4) {
        printf("use %s arraysize streamlen nworkers\n", argv[0]);
        return -1;
    }

    size_t inputsize   =atol(argv[1]);
    long   streamlen   =atoi(argv[2]);
    int    nworkers    =atoi(argv[3]);

    ff_farm<> farm;
    Emitter   E(streamlen,inputsize);
    Collector C;
    farm.add_emitter(&E);
    farm.add_collector(&C);

    oclTask oclt;
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) 
        w.push_back(new ff_reduceOCL<myTask, oclTask>(reducef));
    farm.add_workers(w);
    farm.cleanup_workers();
    farm.run_and_wait_end();

    printf("DONE\n");
    return 0;
}
