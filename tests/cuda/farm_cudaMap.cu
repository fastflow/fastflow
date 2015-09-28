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
/* Author: Maurizio Drocco
 *         drocco@di.unito.it
 */


#if !defined(FF_CUDA)
#define FF_CUDA
#endif

#include <ff/farm.hpp>
#include <ff/stencilReduceCUDA.hpp>
using namespace ff;

#include <iostream>
using namespace std;

#ifdef CHECK
int globalerr = 0;
#endif

FFMAPFUNC(mapF, unsigned int, in,
		return in + 1;
);

// this is global just to keep things simple
size_t inputsize;

class cudaTask: public baseCUDATask<unsigned int, unsigned int, unsigned char, unsigned char, int, char, char> {
public:
	void setTask(void* t) {
    	if (t) {
    		cudaTask *t_ = (cudaTask *)t;
    		setInPtr(t_->in);
    		setOutPtr(t_->in);
    		//setOutPtr(t_->out);
    		setSizeIn(inputsize);
    	}
    }

	unsigned int *in, *out;
};

class Emitter: public ff_node {
public:
    Emitter(int streamlen, size_t size):streamlen(streamlen),size(size) {}

    void* svc(void*) {
        for(int i=0;i<streamlen;++i) {
        	cudaTask *task = new cudaTask();
            task->in = new unsigned int[size];
            //task->out = new unsigned int[size];
            for(size_t j=0;j<size;++j)
            	task->in[j]=j;
            ff_send_out(task);
        }
        return NULL;
    }
private:
    int streamlen;
    size_t size;
};


class Collector: public ff_node {
public:
    Collector(size_t size):size(size) {}

    void* svc(void* t) {
        cudaTask* task = (cudaTask*)t;
#if defined(CHECK)
//        for(long i=0;i<size;++i)  printf("%d ", task->in[i]);
//        printf("\n");
        unsigned int sum = 0;
        for(size_t i=0; i<size; ++i)
        	sum += task->in[i];
        globalerr |= (sum != (size * (size + 1)) / 2);
#endif
        delete task;
        return GO_ON;
    }
private:
    size_t size;
};


int main(int argc, char * argv[]) {
	inputsize = 2048;
	int streamlen = 128;
	int nworkers = 2;

    if(argc>1)inputsize=atoi(argv[1]);
    if(argc>2)streamlen=atoi(argv[2]);
    if(argc>3)nworkers =atoi(argv[3]);

    printf("using arraysize=%lu streamlen=%lu nworkers=%lu\n", inputsize, streamlen, nworkers);

    ff_farm<> farm;
    Emitter   E(streamlen,inputsize);
    Collector C(inputsize);
    farm.add_emitter(&E);
    farm.add_collector(&C);

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i)
        w.push_back(new FFMAPCUDA(cudaTask, mapF)());
    farm.add_workers(w);
    farm.run_and_wait_end();

#ifdef CHECK
    if(globalerr) {
    	printf("ERROR\n");
    	return 1;
    }
#endif
    printf("DONE\n");
    return 0;
}
