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
 */


#if !defined(FF_OCL)
#define FF_OCL
#endif

#include <ff/map.hpp>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>

using namespace ff;

FFMAPFUNC(mapf, float, elem, return (elem+1.0) );

template<typename T>
class oclTask: public baseTask {
public:
    typedef T base_type;

    oclTask():s(0) {}
    oclTask(base_type* t, size_t s):baseTask(t),s(s) {}

    void   setTask(void* t) { 
        if (t) {  
            task=((oclTask<T>*)t)->getInPtr(); 
            s   =((oclTask<T>*)t)->size();
        } 
    }
    size_t size() const     { return s;} 
    size_t bytesize() const { return s*sizeof(base_type); }    
protected:
    size_t s; 
};




class ArrayGenerator: public ff_node {
public:
    ArrayGenerator(int streamlen, size_t size):
        streamlen(streamlen),size(size) {}

    void* svc(void*) {
        for(int i=0;i<streamlen;++i) {
            float *t = new float[size];
            for(size_t j=0;j<size;++j) t[j]=j+i;
            oclTask<float>* task = new oclTask<float>(t,size);
            ff_send_out((void*)task);
        }
        return NULL;
    }
private:
    int streamlen;
    size_t size;
};

class ArrayGatherer: public ff_node {
public:
    void* svc(void* t) {
        oclTask<float>* task = (oclTask<float>*)t;
        size_t size = task->size();
        float* res = (float*)task->getInPtr();
#if defined(CHECK)
        for(long i=0;i<size;++i)  printf("%.2f ", res[i]);
        printf("\n");
#endif
        delete res; delete task;
        return GO_ON;
    }
};

int main(int argc, char * argv[]) {
    if (argc<4) {
        printf("use %s arraysize streamlen nworkers\n", argv[0]);
        return -1;
    }

    int    size     =atoi(argv[1]);
    int    streamlen=atoi(argv[2]);
    int    nworkers =atoi(argv[3]);

    ff_pipeline pipe;
    pipe.add_stage(new ArrayGenerator(streamlen, size));
    ff_farm<> farm;
    farm.add_collector(NULL);
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i)
        w.push_back(NEWMAPONSTREAM(oclTask<float>, mapf));
    farm.add_workers(w);
    pipe.add_stage(&farm);
    pipe.add_stage(new ArrayGatherer);
    pipe.run_and_wait_end();
    
    printf("DONE\n");
    return 0;
}
