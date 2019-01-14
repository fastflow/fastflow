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
/* Author  : Massimo
 * Date    : April 2014
 * Modified: July 2018 
 */                                    
/*
 *                             
 *        -----                                -----     
 *       |     |                              |     |    
 *       |  W1 | ----                     ----|  W2 | ---
 *        -----      |                   |     -----     |
 *          .        |       -----       |       .       |      -----
 *          .        |----> |     |      |       .       |     |     |
 *          .        |      |  E  | -----|       .       |---->|  C  |
 *        -----      |       -----       |     -----     |      -----
 *       |     |-----          ^          ----|     |----         |
 *       |  W1 |               |              |  W2 |             |
 *        -----                |               -----              |
 *                              ----------------------------------
 *
 *   2-stage pipeline: 
 *     - the first stage is a farm with no Emitter and Collector
 *     - the second stage is a farm with the Collector and feedback channel
 *     - the Collector C uses the all_gather method
 *
 */

#include <ff/ff.hpp>

using namespace ff;
long   NUMTASKS=10;
int    FARM1WORKERS=2; 


struct W1: ff_node_t<long> {
    long* svc(long*) {
        for(long i=(get_my_id()+1); i<=NUMTASKS; ++i) {
            ff_send_out((long*)i);
        }
        return EOS;
    }
};

struct W2: ff_node_t<long> {
    long* svc(long* task) {
        long t = (long)task;
        assert(t>1);
        --t;
        return (long*)t;
    }
};

class E: public ff_monode_t<long> {
public:
    long *svc(long *task) {
        long t = (long)task;
        if (get_channel_id() == -1) { // message coming from the input channels
            if (t == 1) return GO_ON;            
            ++numtasks;
            printf("INPUT: sending %ld to all workers\n", t);
            broadcast_task(task);
            return GO_ON;
        }
        printf("BACK: got  %ld from collector (numtasks=%ld)\n", t,numtasks);
        
        if (t != 1) {
            broadcast_task(task);
            return GO_ON;
        }
        if (--numtasks == 0) return EOS;
        return GO_ON;
    }
protected:
    int neos=0;
    long numtasks=0;
};

struct C: ff_minode_t<long> {
    long* svc(long* task) {
        std::vector<long*> V;        
        all_gather(task, V);
        return V[0];
    }
};


int main(int argc, char *argv[]) {
    if (argc > 1) {
        NUMTASKS=atol(argv[1]);
    }

    std::vector<std::unique_ptr<ff_node>> w1;
    for(int i=0;i<FARM1WORKERS;++i)
        w1.push_back(make_unique<W1>());
    ff_Farm<> farm1(std::move(w1));
    farm1.remove_collector();
    
    E e;
    C c;
    std::vector<std::unique_ptr<ff_node>> w2;
    for(int i=0;i<FARM1WORKERS;++i)
        w2.push_back(make_unique<W2>());
    ff_Farm<> farm2(std::move(w2),e, c);

    farm2.wrap_around();
    ff_pipeline pipe;
    pipe.add_stage(&farm1);
    pipe.add_stage(&farm2);

    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }
    printf("DONE\n");
    return 0;
}
