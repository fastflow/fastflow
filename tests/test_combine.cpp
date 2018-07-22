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
 *
 *
 */
/* Author: Massimo Torquati
 *
 */
// basic composition of sequential stages

#include <iostream>

#include <ff/config.hpp>
#include <ff/pipeline.hpp>
#include <ff/combine.hpp>

using namespace ff;

long ntasks = 1000;
long worktime = 5; // usecs

struct firstStage: ff_node_t<long> {
    long *svc(long*) {
        for(long i=1;i<=ntasks;++i) {
            ff_send_out((long*)i);
        }
        return EOS; // End-Of-Stream
    }
};
struct secondStage: ff_node_t<long> { // 2nd stage
    long *svc(long *t) {
        usleep(worktime);
        return t;
    }
};
struct secondStage2: ff_node_t<long> { // 2nd stage
    long *svc(long *t) {
        usleep(worktime);
        return t;
    }
};
struct secondStage3: ff_node_t<long> { // 2nd stage
    long *svc(long *t) {
        usleep(worktime);
        return t;
    }
};
struct thirdStage: ff_node_t<long> {  // 3rd stage
    long *svc(long *task) {
        usleep(worktime);
        return GO_ON; 
    }
}; 
int main() {
    firstStage    _1;
    secondStage   _2;
    secondStage2  _3; 
    secondStage3  _4;
    thirdStage    _5;
    
    unsigned long inizio=getusec();
    long *r;
    for(long i=1;i<=ntasks;++i) {
        r = _2.svc((long*)i);
        r = _3.svc(r);
        r = _4.svc(r);
        r = _4.svc(r);
    }
    unsigned long fine=getusec();
    std::cout << "TEST  FOR  Time = " << (fine-inizio) / 1000.0 << " ms\n";

    ff_Pipe<> pipe(_1,_2,_3,_4,_5);
    pipe.run_and_wait_end();
    std::cout << "TEST  PIPE Time = " << pipe.ffwTime() << " ms\n";
    
    {    
    // we declared them const because we want to re-use the same nodes in multiple tests
    const firstStage    _1;
    const secondStage   _2;
    const secondStage2  _3; 
    const secondStage3  _4;
    const thirdStage    _5;
    {
        ff_Pipe<> pipe(_1, combine_nodes(
                                     combine_nodes(_2, _3),
                                     combine_nodes(_4, _5)
                                     ));
        // starts the pipeline
        if (pipe.run_and_wait_end()<0)
            error("running pipe\n");
        std::cout << "TEST0 DONE Time = " << pipe.ffwTime() << " ms\n";
    }
    usleep(500000);
    {
        ff_Pipe<> pipe(_1, combine_nodes(combine_nodes(_2, _3), combine_nodes(_4, _5)) );
        // starts the pipeline
        if (pipe.run_and_wait_end()<0)
            error("running pipe\n");
        std::cout << "TEST1 DONE Time = " << pipe.ffwTime() << " ms\n";
    }
    usleep(500000);
    {
        ff_Pipe<> pipe(_1, _2, combine_nodes(_3, combine_nodes(_4, _5)) );
        // starts the pipeline
        if (pipe.run_and_wait_end()<0)
            error("running pipe\n");
        std::cout << "TEST2 DONE Time = " << pipe.ffwTime() << " ms\n";
    }
    usleep(500000);
    {
        ff_Pipe<> pipe(_1, _2, _3, combine_nodes(_4, _5) );
        // starts the pipeline
        if (pipe.run_and_wait_end()<0)
            error("running pipe\n");
        std::cout << "TEST3 DONE Time = " << pipe.ffwTime() << " ms\n";
    }
    usleep(500000);
    {
        ff_Pipe<> pipe( combine_nodes(_1,_2), _3, combine_nodes(_4, _5) );
        // starts the pipeline
        if (pipe.run_and_wait_end()<0)
            error("running pipe\n");
        std::cout << "TEST4 DONE Time = " << pipe.ffwTime() << " ms\n";
    }
    usleep(500000);
    {
        ff_Pipe<> pipe( combine_nodes(combine_nodes(_1,_2), _3), combine_nodes(_4,_5) );
        // starts the pipeline
        if (pipe.run_and_wait_end()<0)
            error("running pipe\n");
        std::cout << "TEST5 DONE Time = " << pipe.ffwTime() << " ms\n";
    }
    usleep(500000);
    {
        ff_Pipe<> pipe( combine_nodes(_1,_2), combine_nodes(_3,_4), _5 );
        // starts the pipeline
        if (pipe.run_and_wait_end()<0)
            error("running pipe\n");
        std::cout << "TEST6 DONE Time = " << pipe.ffwTime() << " ms\n";
    }
    usleep(500000);
    {
        ff_Pipe<> pipe( combine_nodes(combine_nodes(_1,_2), combine_nodes(_3,_4)), _5 );
        // starts the pipeline
        if (pipe.run_and_wait_end()<0)
            error("running pipe\n");
        std::cout << "TEST7 DONE Time = " << pipe.ffwTime() << " ms\n";
    }
    usleep(500000);
    {
        ff_Pipe<> pipe( combine_nodes(combine_nodes(_1,_2), combine_nodes(_3,_4)), _5 );
        // starts the pipeline
        if (pipe.run_and_wait_end()<0)
            error("running pipe\n");
        std::cout << "TEST8 DONE Time = " << pipe.ffwTime() << " ms\n";
    }
    usleep(500000);
    {  
        ff_Pipe<> pipe(  combine_nodes(
                                 combine_nodes(_1,_2),
                                 combine_nodes(_3,_4)
                                 ),
                         _5);
        // starts the pipeline
        if (pipe.run_and_wait_end()<0)
            error("running pipe\n");
        std::cout << "TEST9 DONE Time = " << pipe.ffwTime() << " ms\n";
    }
    usleep(500000);
    }
  
  return 0;
}
