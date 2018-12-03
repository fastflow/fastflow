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

/* simple torus pipeline */
#include <iostream> // cerr
#include <ff/ff.hpp>

using namespace ff;

// some globals
const long NUMTASKS=1048576;
const long bigbatchSize=1024;
const long smallbatchSize=256; // 64
unsigned int nticks=0;
long numBatch;

struct masterStage: public ff_node {
    int svc_init() { eossent=false; return 0;}
    void * svc(void * task) {
        if (task==NULL) {
            for(long i=0;i<bigbatchSize;++i)
                if (!ff_send_out((void*)(i+1))) abort();
            //if (numBatch>0) --numBatch;
            return GO_ON;
        }
        
        if (numBatch) {
            for(long i=0;i<smallbatchSize;++i)
                if (!ff_send_out((void*)(i+1))) abort();
            --numBatch;
        } else if (!eossent) {
            ff_send_out(EOS);
            eossent=true;
        }
        ticks_wait(nticks);
        return GO_ON;
    };
    bool eossent;
};

// all other stages
struct Stage: public ff_node {
    void * svc(void * task) {  ticks_wait(nticks); return task; }
};

//
void usage(char * name) {
    std::cerr << "usage: \n";
    std::cerr << "      " << name << " num-stages nticks\n";
}

int main(int argc, char * argv[]) {
    unsigned int nstages = 7;
    nticks = 1000;
    if (argc>1) {
        if (argc!=3) {
            usage(argv[0]);
            return -1;
        }
        
        nstages  = atoi(argv[1]);
        nticks   = atoi(argv[2]);
    }

    numBatch=((NUMTASKS-bigbatchSize)/smallbatchSize);
    
    if (nstages<2) {
        std::cerr << "invalid number of stages\n";
        return -1;
    }
    
    ff_pipeline pipe(false,512,512,true);
    
    pipe.add_stage(new masterStage);
    for(unsigned i=1;i<nstages;++i)
        pipe.add_stage(new Stage);
    pipe.wrap_around();
    
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    double time= pipe.ffTime();
    printf("Time: %g (ms)  Throughput: %f msg/s\n", time,(NUMTASKS*nstages*1000.0)/time);
    return 0;
}
