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
 *           3-stages pipeline
 *               
 *             --> Worker1 -->         
 *            |               |        
 * Start ----> --> Worker1 -->  -----> Stop
 *            |               |        
 *             --> Worker1 -->         
 *                                     
 *                 farm
 *               
 * This test shows how to implement an ordered farm. In this case
 * the farm respects the FIFO ordering of tasks.....
 *
 *     
 */


#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>

using namespace ff;

class Start: public ff_node_t<long> {
public:
    Start(int streamlen):streamlen(streamlen) {}
    long* svc(long*) {    
        for (int j=0;j<streamlen;j++) {
            ff_send_out(new long(j));
        }
        return NULL;
    }
private:
    int streamlen;
};


// worker function
long *Fworker(long *task, ff_node*const) {
        usleep(random() % 20000);
        return task;
}


class Stop: public ff_node_t<long> {
public:
    Stop():expected(0),error(false) {}

    long* svc(long* t) {    
        long &task = *t;
        if (task != expected) {
            printf("ERROR: task received out of order, received %ld expected %ld\n", task, expected);
            error = true;
        }
        
        expected++;
        return GO_ON;
    }
    
    void svc_end() {
        if (error) abort();
    }

private:
    long expected;
    bool error;
};


int main(int argc, char * argv[]) {
    int nworkers = 3;
    int streamlen = 1000;
    if (argc>1) {
        if (argc<3) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nworkers streamlen\n";
            return -1;
        }
               
        nworkers=atoi(argv[1]);
        streamlen=atoi(argv[2]);
    }

    if (nworkers<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
    srandom(131071);
        
#if 0
    // creates the pipeline adding the first stage and then adding all other stages
    ff_Pipe<> pipe(make_unique<Start>(streamlen));
    pipe.add_stage(make_unique<ff_OFarm<long>>(Fworker, nworkers));
    pipe.add_stage(make_unique<Stop>());
#else  // one-line command
    ff_Pipe<> pipe(make_unique<Start>(streamlen),
                   make_unique<ff_OFarm<long>>(Fworker, nworkers),
                   make_unique<Stop>());
#endif

    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }

    std::cerr << "DONE\n";
    return 0;
}
