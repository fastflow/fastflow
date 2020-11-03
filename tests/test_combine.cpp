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
#include <ff/ff.hpp>
using namespace ff;

long ntasks = 100000;
long worktime = 1*2400; // usecs

#if 1 // two different ways to implement the first stage
struct firstStage: ff_node_t<long> {
    long *svc(long*) {
        for(long i=1;i<=ntasks;++i) {
            ff_send_out((long*)i);
        }
        return EOS; // End-Of-Stream
    }
};
#else
struct firstStage: ff_node_t<long> {
    long *svc(long*) {
        if (myntasks<=0) return EOS;
        long t = myntasks--;
        return (long*)t;
    }
    long myntasks=ntasks;
};
#endif
struct secondStage: ff_node_t<long> { // 2nd stage
    long *svc(long *t) {
        ticks_wait(worktime);
        return t;
    }
};
struct secondStage2: ff_node_t<long> { // 2nd stage
    long *svc(long *t) {
        ticks_wait(worktime);
        return t;
    }
};
struct secondStage3: ff_node_t<long> { // 2nd stage
    long *svc(long *t) {
        ticks_wait(worktime);
        return t;
    }
};
struct thirdStage: ff_node_t<long> {  // 3rd stage
    long *svc(long *) {
        ticks_wait(worktime);
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
        r = _5.svc(r);
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
            auto comb = combine_nodes(_1, combine_nodes(_2, combine_nodes(_3, combine_nodes(_4, _5))));
            if (comb.run_and_wait_end()<0)
                error("running comb\n");
            std::cout << "TEST0 DONE Time = " << comb.ffwTime() << " ms\n";
        }
        usleep(500000);
        {
            ff_Pipe<> pipe(_1, combine_nodes(
                                             combine_nodes(_2, _3),
                                             combine_nodes(_4, _5)
                                             ));
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
            ff_Pipe<> pipe( _1, combine_nodes(_2,_3), combine_nodes(_4, _5) );
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
            ff_Pipe<> pipe( _1, combine_nodes(_2, combine_nodes(_3,_4)), _5 );
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
            ff_Pipe<> pipe( combine_nodes(combine_nodes(_1,_2), combine_nodes(_3,_4)), _5 );
            // starts the pipeline
            if (pipe.run_and_wait_end()<0)
                error("running pipe\n");
            std::cout << "TEST9 DONE Time = " << pipe.ffwTime() << " ms\n";
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
            std::cout << "TEST10 DONE Time = " << pipe.ffwTime() << " ms\n";
        }
        usleep(500000);
        {
            // testing multi-input + multi-output combined

            struct miStage:ff_minode_t<long> {
                long* svc(long* in) {
                    //printf("MULTI INPUT input channels %ld\n", get_num_inchannels());
                    return in;
                }
                void eosnotify(ssize_t) {
                    //printf("MULTI INPUT %ld, eosnotify\n", get_my_id());
                }
            };
            struct moStage:ff_monode_t<long> {
                long* svc(long* in) {
                    return in;
                }
                void eosnotify(ssize_t) {
                    //printf("MULTI OUTPUT %ld, eosnotify\n", get_my_id());
                }
                
            };
            const miStage mi;
            const moStage mo;
            const ff_comb comb0(mi, mi);
            const ff_comb comb1(mo, mo);

            ff_comb comb(comb1, comb0);
            
            ff_Pipe<> pipe(_1, comb, _5);
            // starts the pipeline
            if (pipe.run_and_wait_end()<0)
                error("running pipe\n");
            std::cout << "TEST11 DONE Time = " << pipe.ffwTime() << " ms\n";
        }
    }
  
    return 0;
}
