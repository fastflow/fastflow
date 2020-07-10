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
 * Testing farms in which the Emitter is multi-output and the Collector 
 * is multi-input
 * 
 *                   |-> Worker->|                     |-> Worker->|
 *  First--> moNode->|           |->miNode --> moNode->|           |->miNode --> Last
 *                   |-> Worker->|                     |-> Worker->|
 *
 *  pipe(First, Farm(Worker), Farm(Worker), Last)
 *
 */
/* Author: Massimo Torquati
 *
 */

#include <string>
#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

struct First: ff_node_t<long> {
    First(const int ntasks):ntasks(ntasks) {}
    long* svc(long*) {
        for(long i=1;i<=ntasks;++i) {
            struct timespec req;
            req.tv_sec = 0;
            req.tv_nsec = 5000;
            nanosleep(&req, (struct timespec *)NULL);

            ff_send_out((long*)i);
        }
        return EOS;
    }
    const int ntasks;
};
struct Emitter: ff_monode_t<long> {
    long* svc(long*in) {
        return in;
    }
};
struct Collector: ff_minode_t<long> {
    long* svc(long*in) {
        return in;
    }
};

struct Worker: ff_node_t<long> {
    long* svc(long*in) {
        struct timespec req;
        req.tv_sec = 0;
        req.tv_nsec = 3000*get_my_id();
        nanosleep(&req, (struct timespec *)NULL);        
        return in;
    }
};
struct Last: ff_node_t<long> {
    long* svc(long*) {
        ++counter;
        return GO_ON;
    }
    size_t counter=0;
};


int main(int argc, char* argv[]) {

    // default arguments
    size_t ntasks    = 10000;
    size_t nworkers1 = 3;
    size_t nworkers2 = 2;


    if (argc>1) {
        if (argc!=4) {
            error("use: %s ntasks nworkers1 nworkers2\n",argv[0]);
            return -1;
        }
        ntasks    = std::stol(argv[1]);
        nworkers1 = std::stol(argv[2]);
        nworkers2 = std::stol(argv[3]);
    }

    First first(ntasks);
    Last last;

    /* all the following farms have nworkers1 workers */
    ff_Farm<long,long> farm1([&]() {
                                 std::vector<std::unique_ptr<ff_node> > V;
                                 for(size_t i=0;i<nworkers1;++i)
                                     V.push_back(make_unique<Worker>());
                                 return V;
                             } (),
        make_unique<Emitter>(),
        make_unique<Collector>());

    /* the following farms have nworkers2 workers */
    ff_Farm<long,long> farm2([&]() {
                                 std::vector<std::unique_ptr<ff_node> > V;
                                 for(size_t i=0;i<nworkers2;++i)
                                     V.push_back(make_unique<Worker>());
                                 return V;
                             } (),
        make_unique<Emitter>(),
        make_unique<Collector>());
    
    //ff_Pipe<> pipe(first, farm1, farm2, last);
    ff_Pipe<> pipe(first, farm1, last);
    // running the optimized pipe
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    printf("test DONE\n");
    return 0;
}
