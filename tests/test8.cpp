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
 * Author: Massimo Torquati <torquati@di.unipi.it> 
 * Date:   November 2014
 */

//
// simple 3-stage pipeline using the ff_node_t node interface
// testing the re-definition of global new and delete
//

#include <iostream>
#include <new>
#include <ff/ff.hpp>
// if you want to test the program with valgrind please enable the following define
//#if !defined(MAKE_VALGRIND_HAPPY)
//#define MAKE_VALGRIND_HAPPY 1
//#endif
#include <ff/allocator.hpp>
using namespace ff;

// global new and delete redefined
void * operator new(size_t size) { 
    return ff_malloc(size);
}
void operator delete(void* p) noexcept {
    return ff_free(p);
}
void operator delete(void* p, size_t) noexcept {
    return ff_free(p);
}

typedef std::vector<float> Task;
struct firstStage: ff_node_t<Task> {
    firstStage(const size_t length):length(length) {}
    Task* svc(Task*) {
        for(size_t i=1; i<=length; ++i) {
            Task *t = new Task(i);
            for(size_t j=0;j<t->size();++j) t->operator[](j) = j;
            ff_send_out(t);
        }
        return EOS;
    }
    const size_t length;
};
struct secondStage: ff_node_t<Task> {
    Task* svc(Task* task) { 
        Task &t = *task; 
        for(size_t j=0;j<t.size();++j)
            t[j] *= t[j];
        return task; 
    }
};
struct thirdStage: ff_node_t<Task> {   
    Task* svc(Task* task) { 
        Task &t = *task; 
        for(size_t j=0;j<t.size();++j)
            sum += t[j];
        delete task;
        return GO_ON; 
    }
    void svc_end() { std::cout << "sum = " << sum << "\n"; }
    float sum = 0.0;
};

int main(int argc, char *argv[]) {    
    long streamlen = 500;
    if (argc>1) {
        if (argc!=2) {
            std::cerr << "use: "  << argv[0] << " streamlen\n";
            return -1;
        }
        streamlen=atol(argv[1]);
    }
    firstStage  F1(streamlen);
    secondStage F2;
    thirdStage  F3;
    ff_Pipe<> pipe(F1,F2,F3);
    pipe.setXNodeInputQueueLength(10,true);
    pipe.run_and_wait_end();
    std::cout << "Time: " << pipe.ffTime() << "\n";
    return 0;
}
