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

/* Author: Massimo Torquati
 * Date  : December 2014
 *         
 */

#include <iostream>
#include <cstdio>
#include <string>
#include <memory>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
using namespace ff;

int main() {
    // NOTE: 
    // ff_Farm and ff_Pipe are the high-level task-farm and pipeline patterns built on top of 
    // ff_farm and ff_pipeline core patterns.
    // They have a different interface with respect to ff_farm and ff_pipe "low-level" patterns.
    // 
    //
    // MAIN DIFFERENCES:
    // ff_node_F: It creates FastFlow node from a function.
    //
    // ff_Pipe<IN_t, OUT_t>: 
    //          It accepts stages having different input and output types provided that
    //          the correct type ordering is respected, otherwise a compilation error is raised (static_assert).
    //          It is a template class whose template types IN_t, OUT_t are the input and output type of the 
    //          entire pipeline, respectively. 
    //          If ff_Pipe<> is used (i.e. withouth IN_t OUT_t) then it means that the pipeline will not be 
    //          connected to any previous stage for receiving inputs and its outputs are discarded by the last stage.
    //        
    // ff_Farm<IN_t, OUT_t> : 
    //          It accepts a vector of workers or a function/lambda having a proper signature.
    //          It is a template class whose template types IN_t, OUT_t are the input and output type of the 
    //          entire farm, respectively. 
    //          The default Emitter and Collector are added by default.
    //          Workers are passed as unique_ptr(s) so the worker object will be deleted in the ff_Farm destructor. 
    //          The Emitter and Collector can be passed either as unique_ptr or as reference. 
    //          In the first case the 2 objects will be deleted with the destruction of the ff_Farm.
    //          if ff_Farm<> is used then it means that the farm will not be connected with in input and output.

    {  	// ------------------------- example 1 ---------------------------------

        static std::string ExNumber = "1";

        // generates 10 tasks
        struct Stage0: ff_node_t<char, long> {  // NOTE: 'void' cannot be used as input/output type!
            long *svc(char *) {
                for(size_t i=0;i<10;++i)
                    ff_send_out(new long(i));
                return EOS;
            }
        };	
        // prints tasks
        struct Stage1: ff_node_t<long,char> {   // NOTE: 'void' cannot be used as input/output type!
            int svc_init() { std::cout << "Ex " << ExNumber << ":\n"; return 0;}
            char *svc(long *task) {
                std::cout << *task << " ";
                delete task;
                return GO_ON;
            }
            void svc_end() { std::cout << "\n----------\n"; }
        };
        // function: it adds 1 to the input task and then returns the task
        auto F = [](long *in, ff_node*const)->long* {
            *in += 1;
            return in;
        };
		
        Stage0 s0;
        Stage1 s1;
        
        // creating stages from a function F
        ff_node_F<long,long> s01(F), s011(F);
        
        // creates a farm from a function F with 3 workers, 
        // default emitter and collector implicitly added to the farm
        ff_Farm<long,long> farm(F,3);
        //          ^
        //          |----- here the types have to be set because the 'farm' will be used in a pipeline (see below). 
        
        // pipe with a farm
        ff_Pipe<long,long> pipe0(s01,s011, farm); // NOTE: references used here instead of pointers
        //          ^
        //          |----- here the types have to be set because the 'pipe0' will be used as a stage of another pipeline 
        
        // main pipe, no input and output stream
        ff_Pipe<> pipe(s0, pipe0, s1);      // NOTE: references used here instead of pointers
        
        if (pipe.run_and_wait_end()<0) error("running pipe\n");
        
        // ------------------------- example 2 ---------------------------------
        
        // -- option 1 --
#if 1
        // NOTE: all workers and emitter and collector will be automatically deleted 
        ff_Farm<> farm2([F]() {
                std::vector<std::unique_ptr<ff_node> > W;	
                for(size_t i=0;i<3;++i)
                    W.push_back(std::unique_ptr<ff_node_F<long> >(make_unique<ff_node_F<long>>(F)));
                return W;
            } (), 
            std::unique_ptr<ff_node>(make_unique<Stage0>()), 
            std::unique_ptr<ff_node>(make_unique<Stage1>()));
        
#else   // -- option 2 --

        // NOTE: vectors of base ff_node.  unique_ptr is used here !
        std::vector<std::unique_ptr<ff_node> > W;	
        for(size_t i=0;i<3;++i)
            W.push_back(std::unique_ptr<ff_node_F<long> >(make_unique<ff_node_F<long>>(F)));
               
        // NOTE: all workers will be automatically deleted 
        ff_Farm<> farm2(std::move(W), s0, s1);
#endif
        ExNumber = "2";
        if (farm2.run_and_wait_end()<0) error("running farm\n");
    }
    
    {  	// ------------------------- example 3 ---------------------------------
        
        /*
         *                          | --- W ---|
         *              g -----> E -| --- W ---| - C ------> d
         *                          | --- W ---|
         * stream types:   long    string    long    string   
         *
         */
        struct Generator: ff_node_t<char, long> {
            Generator(size_t N):N(N) {}
            long *svc(char*) {
                for(size_t i=0;i<N;++i)
                    ff_send_out(new long(i));
                return EOS;
            };
            const size_t N;
        };
        struct Emitter: ff_node_t<long, std::string> {
            std::string* svc(long *t) {
                std::string *st = new std::string;
                *st = std::to_string(*t);
                delete t;
                return st;
                
            }
        };
        struct Worker: ff_node_t<std::string, long> {
            long *svc(std::string *t) {
                long x = atol(t->c_str());
                delete t;
                return new long(x);
            }
        };
        struct Collector: ff_node_t<long, std::string> {
            std::string* svc(long *t) {
                std::string *st = new std::string;
                *st = std::to_string(*t);
                delete t;
                return st;	    
            }
        };
        struct Drainer: ff_node_t<std::string, char> {
            int svc_init() { std::cout << "Ex 3:\n"; return 0;}	    
            char *svc(std::string *t) {
                std::cout << "received " << *t << "\n";
                delete t;
                return GO_ON;
            }
            void svc_end() { std::cout << "\n----------\n"; }
        };
        
        std::vector<std::unique_ptr<ff_node> > W;	
        for(size_t i=0;i<3;++i)
            W.push_back(std::unique_ptr<ff_node_t<std::string, long> >(make_unique<Worker>()));
        
        Emitter e;
        Collector c;
        
        ff_Farm<long, std::string> farm(std::move(W), e, c);
        
        Generator g(10);
        Drainer   d;
        ff_Pipe<> pipe (g, farm, d);
        
        if (pipe.run_and_wait_end()<0) error("running pipe\n");
    }


    {  	// ------------------------- example 4 ---------------------------------
        
        /*                       
         *                          | --- W ---|   
         *              g -----> S -| --- W ---| --|
         *                       ^  | --- W ---|   |
         *                       |                 |
         *                        -----------------
         */
        struct Generator: ff_node_t<long> {
            Generator(size_t N):N(N) {}
            int svc_init() { std::cout << "Ex 4:\n"; return 0;}	    
            long *svc(long*) {
                for(size_t i=0;i<N;++i)
                    ff_send_out(new long(i));
                return EOS;
            };
            const size_t N;
        };
        struct Scheduler: ff_node_t<long> {
            ff_loadbalancer* lb;
            long *svc(long *t) {
                if (lb->get_channel_id() == -1) return t;
                delete t;
                return GO_ON;
            }            
            void eosnotify(ssize_t id) {
                if (id==-1) lb->broadcast_task(EOS);
            }
            void svc_end() { std::cout << "\n----------\n"; }
            void setLB(ff_loadbalancer *lb_) { lb=lb_;}
        };
        struct Worker: ff_node_t<long> {
            long *svc(long *t) { 
                std::cout << "W" << get_my_id() << ": got " << *t << "\n";
                return t; 
            }
        };

        std::vector<std::unique_ptr<ff_node> > W;	
        for(size_t i=0;i<3;++i)
            W.push_back(std::unique_ptr<ff_node_t<long,long> >(make_unique<Worker>()));
        
        Scheduler S;
        ff_Farm<long> farm(std::move(W), S);
        farm.remove_collector(); // remove default collector
        farm.wrap_around();
        S.setLB(farm.getlb());

        Generator g(10);
        ff_Pipe<> pipe (g, farm);
        
        if (pipe.run_and_wait_end()<0) error("running pipe\n");
    }

    return 0;
}
