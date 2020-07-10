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
/*
 *
 * |<----------------------- farm ------------------->|            
 *                    |<---- all-to-all ---->|
 * 
 *             ------> Router-->
 *             |                |---> Even ->
 *             |                |            |
 * Generator --|-----> Router-->|            | ---> Sink
 *             |                |            |
 *             |                |--->  Odd ->
 *             ------> Router-->
 *
 * 
 * NOTE: another version can be obtained by using all building-blocks:
 *     pipe(Generator, A2A(Router, Even), Sink)
 *
 */
/* Author: Massimo Torquati
 *
 */

#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

struct Generator: ff_node_t<long> { 
    long *svc(long*) {
        // sum: odd 271 even 344
        ff_send_out(new long(13));
        ff_send_out(new long(17));
        ff_send_out(new long(14));       
        ff_send_out(new long(27));
        ff_send_out(new long(31));
        ff_send_out(new long(6));
        ff_send_out(new long(8));
        ff_send_out(new long(4));
        ff_send_out(new long(26));
        ff_send_out(new long(31));
        ff_send_out(new long(105));
        ff_send_out(new long(238));
        ff_send_out(new long(47));
        ff_send_out(new long(48));
        return EOS; // End-Of-Stream
    }
};
struct Sink: ff_node_t<long> {  
    long *svc(long *task) {
        std::cout <<  *task << "\n";
        delete task;
        return GO_ON; 
    }
}; 

struct Router: ff_monode_t<long> {
    long *svc(long *in) {

        printf("Router%ld received %ld\n", get_my_id(), *in);

        if ((*in % 2) == 0) {
            ff_send_out_to(in, 0);
        } else
            ff_send_out_to(in, 1);

        return GO_ON;
    }
};

struct Even: ff_node_t<long> {
	long *svc(long *in) {
        sum+=*in;
        delete in;
	    return GO_ON;
    }
    void eosnotify(ssize_t=-1) {
        printf("Even EOS received\n");
        ff_send_out(new long(sum));
    }

    long sum=0;
};
struct Odd: ff_node_t<long> {
	long *svc(long *in) {
        sum+=*in;
        delete in;
	    return GO_ON;
    }
    void eosnotify(ssize_t=-1) {
        printf("Odd EOS received\n");
        ff_send_out(new long(sum));
    }

    long sum=0;
};

struct doNothing: ff_node_t<long> {
    long *svc(long *in) {
        return in;
    }
};


int main() {

    std::vector<ff_node*> W1;  
    Router w1, w2, w3;
    W1.push_back(&w1);
    W1.push_back(&w2);
    W1.push_back(&w3);

    std::vector<ff_node*> W2;
    Even even;
    Odd  odd;
    W2.push_back(&even);
    W2.push_back(&odd);

    /* NOTE: if we want to reduce the number of 
     *       channels in the all-to-all building block
     *       we have to set 'true' as first parameter 
     *       of the constructor.
     */
    ff_a2a a2a;
    a2a.add_firstset(W1);
    a2a.add_secondset(W2);

    std::vector<ff_node* > W;    
#if 1
    // to test combine_with_firststage
    ff_pipeline fakepipe;
    fakepipe.add_stage(&a2a);
    combine_with_firststage(fakepipe, new doNothing, true);
    
    W.push_back(&fakepipe);    
#else
    W.push_back(&a2a);
#endif

    Generator gen;
    Sink      sink;
    ff_farm farm(W, &gen, &sink);        
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    
    return 0;
}
