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
 *  testing the helper function 'combine_nodes_in_pipeline'
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
            req.tv_nsec = 3000;
            nanosleep(&req, (struct timespec *)NULL);

            ff_send_out((long*)i);
        }
        return EOS;
    }
    const int ntasks;
};
struct Worker: ff_node_t<long> {
    long* svc(long*in) {
        switch(get_my_id()) {
        case 0: {
            struct timespec req;
            req.tv_sec = 0;
            req.tv_nsec = 50000;
            nanosleep(&req, (struct timespec *)NULL);
        } break;
        case 2: {
            struct timespec req;
            req.tv_sec = 0;
            req.tv_nsec = 20000;
            nanosleep(&req, (struct timespec *)NULL);
        } break;
        default: ; // zero work for the others
        }
        return in;
    }
};
struct Last: ff_node_t<long> {
    long* svc(long*in) {
        printf("Last received %ld\n", (long)in);
        ++counter;
        return GO_ON;
    }
    size_t counter=0;
};
struct Emitter: ff_monode_t<long> {
    long *svc(long *in) {
        ff_send_out(in);
        return GO_ON;
    }
};
struct Collector: ff_minode_t<long> {
    long *svc(long *in) {
        //printf("Collector received from %ld\n", get_channel_id());
        return in;
    }
};


int main(int argc, char* argv[]) {
    // default arguments
    size_t ntasks    = 10000;
    size_t nworkers  = 4;

    if (argc>1) {
        if (argc!=3) {
            error("use: %s ntasks nworkers\n",argv[0]);
            return -1;
        }
        ntasks    = std::stol(argv[1]);
        nworkers  = std::stol(argv[2]);
    }
    { // two sequential
        First first(ntasks);
        Last last;
        auto pipe = combine_nodes_in_pipeline(first,last);
        if (pipe.run_and_wait_end()<0) {
            error("running pipeline\n");
            return -1;
        }
        if (last.counter!=ntasks) {
            error("TEST FAILED (%ld)\n", last.counter);
            return -1;
        }
    }
    printf("TEST1 DONE\n");
    usleep(500000);

    { // merging first stage with farm's emitter
        First first(ntasks);
        Last last;
        Emitter E;
        Collector C;
        ff_Farm<long,long> farm([&]() {
                std::vector<std::unique_ptr<ff_node> > V;
                for(size_t i=0;i<nworkers;++i)
                    V.push_back(make_unique<Worker>());
                return V;
            } (), E, C);        
        auto pipe1 = combine_nodes_in_pipeline(first,farm);
        ff_Pipe<> pipe(pipe1, last);
        if (pipe.run_and_wait_end()<0) {
            error("running pipeline\n");
            return -1;
        }
        if (last.counter!=ntasks) {
            error("TEST FAILED (%ld)\n", last.counter);
            return -1;
        }
    }
    printf("TEST2 DONE\n");
    usleep(500000);
    { // merging first stage with farm's emitter
        First first(ntasks);
        Last last;
        Emitter E;
        Collector C;
        ff_Farm<long,long> farm([&]() {
                std::vector<std::unique_ptr<ff_node> > V;
                for(size_t i=0;i<nworkers;++i)
                    V.push_back(make_unique<Worker>());
                return V;
            } (), E, C);        
        auto pipe1 = combine_nodes_in_pipeline(farm, last);
        ff_Pipe<> pipe(first,pipe1);
        if (pipe.run_and_wait_end()<0) {
            error("running pipeline\n");
            return -1;
        }
        if (last.counter!=ntasks) {
            error("TEST FAILED (%ld)\n", last.counter);
            return -1;
        }
    }    
    printf("TEST3 DONE\n");
    usleep(500000);
    { // merging collector and emitter of the two farms (it isn't the all-to-all introduction) 
        First first(ntasks);
        Last last;
        Emitter E;
        Collector C;
        ff_Farm<long,long> farm1([&]() {
                std::vector<std::unique_ptr<ff_node> > V;
                for(size_t i=0;i<nworkers;++i)
                    V.push_back(make_unique<Worker>());
                return V;
            } ());
        farm1.add_collector(C);
        ff_Farm<long,long> farm2([&]() {
                std::vector<std::unique_ptr<ff_node> > V;
                for(size_t i=0;i<nworkers;++i)
                    V.push_back(make_unique<Worker>());
                return V;
            } (), E);        
        auto pipe1 = combine_nodes_in_pipeline(farm1,farm2);
        ff_Pipe<> pipe(first, pipe1, last);
        if (pipe.run_and_wait_end()<0) {
            error("running pipeline\n");
            return -1;
        }
        if (last.counter!=ntasks) {
            error("TEST FAILED (%ld)\n", last.counter);
            return -1;
        }
    }
    printf("TEST4 DONE\n");
    usleep(500000);
    { // merging collector and emitter of the two ORDERED farms 
        First first(ntasks);
        struct Last: ff_node_t<long> {
            long* svc(long*in) {
                printf("Last received %ld\n", (long)in);
                if ((long)in != expected) {
                    error("Last WRONG ORDERING, received %ld expected %ld\n", (long)in, expected);
                    exit(-1);
                }
                ++counter;
                ++expected;
                return GO_ON;
            }
            size_t counter=0;
            long expected=1;
        } last;                
        Emitter E;
        Collector C;
        ff_OFarm<long,long> farm1([&]() {
                std::vector<std::unique_ptr<ff_node> > V;
                for(size_t i=0;i<nworkers;++i)
                    V.push_back(make_unique<Worker>());
                return V;
            } ());
        farm1.add_collector(C);
        ff_OFarm<long,long> farm2([&]() {
                std::vector<std::unique_ptr<ff_node> > V;
                for(size_t i=0;i<(nworkers+1);++i)   // different n. of workers
                    V.push_back(make_unique<Worker>());
                return V;
            } ());
        farm2.add_emitter(E);
        auto pipe1 = combine_nodes_in_pipeline(farm1,farm2);
        ff_Pipe<> pipe(first, pipe1, last);
        if (pipe.run_and_wait_end()<0) {
            error("running pipeline\n");
            return -1;
        }
        if (last.counter!=ntasks) {
            error("TEST FAILED (%ld)\n", last.counter);
            return -1;
        }
    }
    printf("TEST5 DONE\n");    
    usleep(500000);
    { // everything allocated on the heap
        First *first  = new First(ntasks);
        Last  *last   = new Last;
        Emitter *E    = new Emitter;
        Collector *C  = new Collector;
        ff_farm *farm = new ff_farm;
        std::vector<ff_node* > V;
        for(size_t i=0;i<nworkers;++i) V.push_back(new Worker);
        farm->add_workers(V);
        farm->add_emitter(E);
        farm->add_collector(C);
        farm->cleanup_all();
        const auto pipe1 = combine_nodes_in_pipeline(*farm,*last, true, true);
        ff_pipeline *pipe = new ff_pipeline;         
        pipe->add_stage(first);
        pipe->add_stage(pipe1);
        pipe->cleanup_nodes();
        printf("N. thread=%d\n", pipe->cardinality());        
        if (pipe->run_and_wait_end()<0) {
            error("running pipeline\n");
            return -1;
        }
        if (last->counter!=ntasks) {
            error("TEST FAILED (%ld)\n", last->counter);
            return -1;
        }
        delete pipe;
    }
    printf("TEST6 DONE\n");
    return 0;
}
