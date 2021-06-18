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
 * Author: Massimo Torquati
 * Date:   June 2015
 */

/* Simple torus pipeline each stage sends messages having a size in 
 * [MIN_SIZE; MAX_SIZE[ 
 *
 */

#include <iostream> // cerr
#include <ff/ff.hpp>

using namespace ff;

// some globals
const long NUMTASKS=10485760;
const long bigbatchSize=2048;
const long smallbatchSize=128;
const long MAX_SIZE=32;
const long MIN_SIZE=4;
long numBatch;

// standard memory allocator
#define MALLOC malloc
#define FREE   free

#define MY_RAND_MAX         32767
/* __thread */ unsigned long next = 1; // initial seed should be per thread

inline static double Random(void) {
    next = next * 1103515245 + 12345;
    return (((unsigned) (next/65536) % (MY_RAND_MAX+1)) / (MY_RAND_MAX +1.0));
}

struct masterStage: public ff_node_t<unsigned int> {
    char *prepare_msg() {
        unsigned int size = (unsigned)((Random()*(MAX_SIZE - MIN_SIZE))) + MIN_SIZE;
        char* buf = (char*)MALLOC(sizeof(unsigned int)*(size+1));
        unsigned int *msg = (unsigned int*)buf;
        msg[0] = size;
        msg += 1;
        for (unsigned int k=0; k<size; ++k) msg[k] = counter_out++;
        return buf;
    }
    void check_input(unsigned int *msg) {
        unsigned int size = msg[0];
        if (size < MIN_SIZE || size > MAX_SIZE) {
            fprintf(stderr, "Stage%ld: inconsistent size, exit\n", get_my_id());
            abort();
        }
        ++msg;
        for (unsigned int k=0; k<size; ++k) {
            if (counter_in++ != msg[k]) {
                fprintf(stderr, "Stage%ld: inconsistent data, exit\n", get_my_id()); 
                abort();
            }
        }
    }

    int svc_init() { eossent=false; counter_out=0; counter_in=0; return 0;}
    unsigned int *svc(unsigned int *task) {
        if (task==NULL) {
            // generates a big batch of messages
            for(long i=0;i<bigbatchSize;++i) {
                if (!ff_send_out(prepare_msg())) abort();
            }
            if (numBatch>0) --numBatch;
            return GO_ON;
        }
        
        check_input(task);
        FREE(task);
        if (numBatch) {
            // generates a small batch of messages
            for(long i=0;i<smallbatchSize;++i) 
                if (!ff_send_out(prepare_msg())) abort();
            --numBatch;
        } else if (!eossent) {
            ff_send_out(EOS);
            eossent=true;
        }
        return GO_ON;
    };
    bool eossent;
    unsigned int counter_out;
    unsigned int counter_in;
};

// all other stages
struct Stage: public ff_node_t<unsigned int> {
    void check_input(unsigned int *msg) {
        unsigned int size = msg[0];
        if (size < MIN_SIZE || size > MAX_SIZE) {
            fprintf(stderr, "Stage%ld: inconsistent size, exit\n", get_my_id());
            abort();
        }
        ++msg;
        for (unsigned int k=0; k<size; ++k) {
            if (counter_in++ != msg[k]) {
                fprintf(stderr, "Stage%ld: inconsistent data, exit\n", get_my_id()); 
                abort();
            }
        }
    }

    int svc_init() { counter_in=0; return 0;}
    unsigned int *svc(unsigned int *task) {  
        check_input(task);
        return task; 
    }
    unsigned int counter_in;
};

//
void usage(char * name) {
    std::cerr << "usage: \n";
    std::cerr << "      " << name << " num-stages\n";
}

int main(int argc, char * argv[]) {
    unsigned int nstages = 7;
    if (argc>1) {
        if (argc!=2) {
            usage(argv[0]);
            return -1;
        }
        
        nstages  = atoi(argv[1]);
    }

    numBatch=((NUMTASKS-bigbatchSize)/smallbatchSize);
    
    if (nstages<2) {
        std::cerr << "invalid number of stages\n";
        return -1;
    }
    
    ff_pipeline pipe;
    pipe.add_stage(new masterStage);
    for(unsigned i=1;i<nstages;++i)
        pipe.add_stage(new Stage);
    pipe.wrap_around();
    pipe.setXNodeInputQueueLength(512,false);
    pipe.setXNodeOutputQueueLength(512,false);
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    double time= pipe.ffTime();
    printf("Time: %g (ms)  Throughput: %f msg/s\n", time,(NUMTASKS*nstages*1000.0)/time);
    return 0;
}
