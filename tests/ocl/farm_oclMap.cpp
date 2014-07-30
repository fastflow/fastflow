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


#if !defined(FF_OCL)
#define FF_OCL
#endif

#include <ff/farm.hpp>
#include <ff/map.hpp>

using namespace ff;

FFMAPFUNC(mapf, float, elem, return (elem+1.0) );

// this is gobal just to keep things simple
size_t inputsize=0;

template<typename T>
class oclTask: public baseTask {
protected:
    size_t s; 
public:
    typedef T base_type;

    oclTask():s(0) {}
    oclTask(base_type* t, size_t s):baseTask(t),s(s) {}

    void   setTask(void* t) { if (t) { task=t; s=inputsize; } }
    size_t size() const     { return s;} 
    size_t bytesize() const { return s*sizeof(base_type); }    
#if defined(NOT_INPLACE)
    void*  newOutPtr()    { outPtr= new T[s]; return outPtr; }
    void   deleteOutPtr() { if (outPtr) delete [] outPtr; }
protected:
    T *outPtr;
#endif
};

class Emitter: public ff_node {
public:
    Emitter(int streamlen, size_t size):streamlen(streamlen),size(size) {}

    void* svc(void*) {
        for(int i=0;i<streamlen;++i) {
            float* task = new float[size];
            for(size_t j=0;j<size;++j) task[j]=j+i;
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
        float* task = (float*)t;
#if defined(CHECK)    
        for(long i=0;i<size;++i)  printf("%.2f ", task[i]);
        printf("\n");
#endif

        return GO_ON;
    }
private:
    size_t size;
};


int main(int argc, char * argv[]) {
    if (argc<4) {
        printf("use %s arraysize streamlen nworkers\n", argv[0]);
        return -1;
    }

    inputsize       =atoi(argv[1]);
    int    streamlen=atoi(argv[2]);
    int    nworkers =atoi(argv[3]);

    ff_farm<> farm;
    Emitter   E(streamlen,inputsize);
    Collector C(inputsize);
    farm.add_emitter(&E);
    farm.add_collector(&C);

    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) 
        w.push_back(NEWMAPONSTREAM(oclTask<float>, mapf));
    farm.add_workers(w);
    farm.run_and_wait_end();

    printf("DONE\n");
    return 0;
}
