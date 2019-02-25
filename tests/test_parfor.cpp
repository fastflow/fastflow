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
 *
 */

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>

using namespace ff;

#if defined(ON_DEMAND)
class Ondemand: public ff_node {
public:
    Ondemand(long numtasks):numtasks(numtasks) {}
    void* svc(void*) {
        for(int i=0;i<numtasks;++i) {
            forall_task_t *task = new forall_task_t(i,i+1);
            ff_send_out(task);
        }
        return NULL;
    }
private:
    long numtasks;
};
class W: public ff_node {
public:
    W(const long *V):V(V) {}
    void* svc(void* t) {
        forall_task_t* task = (forall_task_t*)t;
        for(long i=task->start;i<task->end;++i) {
            usleep(V[i]);
        }
        return GO_ON;
    }
private:
    const long *const V;
};
#endif 

int main(int argc, char* argv[]) {
    int nw = 3;
    long size= 500;
    long chunk=100;
    if (argc>1) {
        if (argc < 4) {
            printf("use: %s numworkers size chunk\n", argv[0]);
            return -1;
        }
        nw    =atoi(argv[1]);
        size  =atol(argv[2]); 
        chunk =atol(argv[3]); 
    }
    long *V = new long[size];
    for(int i=0;i<size;++i) {
        V[i] = 1000;
    }
    for(int j=0,i=(size/2-(size/4)); i<(size/2);++i,++j) {
        V[i] += V[i-1];
        V[(size/2+(size/4)-1)-j] = V[i];
    }

    ffTime(START_TIME);
#if 0
    parallel_for(0,size,1,chunk, 
                 [&](const long i) { 
                     usleep(V[i]); 
                 }, nw);
#else
    FF_PARFOR_BEGIN(test1, i,0,size,1, chunk,nw) {
        //printf("I'm thread %d\n", _ff_thread_id);
        usleep(V[i]);
    } FF_PARFOR_END(test1);    
#endif
    ffTime(STOP_TIME);
    printf("Time =%g\n", ffTime(GET_TIME));

#if defined(ON_DEMAND)
    ff_farm   farm;
    std::vector<ff_node *> w;
    farm.add_emitter(new Ondemand(size));
    farm.set_scheduling_ondemand(1);
    for(int i=0;i<nw;++i) w.push_back(new W(V));
    farm.add_workers(w);
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    printf("Time =%g\n", farm.ffTime());
    printf("wTime=%g\n", farm.ffwTime());
#endif
    printf("done\n");
    return 0;
}
