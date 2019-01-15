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

/**
 *
 * Simple test which shows how to implement a MIDS farm with a programmable Emitter.
 *
 * We have 2 kind of workers: Worker1 takes care of even numbers whereas Worker2 of odd ones.
 *
 */
#include <cstdlib>
#include <vector>
#include <ff/ff.hpp>

using namespace ff;

class Worker1: public ff_node {
public:
    void* svc(void* t) {
        const long& task = *(long*)t;
        printf("Worker1 got %ld\n", task);
        assert((task % 2 ) == 0);
        return GO_ON;
    }
};

class Worker2: public ff_node {
public:
    void* svc(void* t) {
        const long& task = *(long*)t;
        printf("Worker2 got %ld\n", task);
        assert((task % 2 ) != 0);	
        return GO_ON;
    }
};


class Emitter: public ff_monode {
public:
    Emitter(int nw1, int nw2, int ntasks):
        nw1(nw1),nw2(nw2),even(0),odd(0),ntasks(ntasks) {}
    
    int svc_init() {
        srandom(::getpid()+(getusec()%4999));
        return 0;
    }
    
    void * svc(void *) {	
        for(int i=0;i<ntasks;++i) {
            long * t = new long(random() % 10485760);
            if (*t % 2)	ff_send_out_to(t, ((odd++%nw2) + nw1));
            else        ff_send_out_to(t, (even++%nw1));	    
        }
        return EOS;
    }    
private:
    int nw1,nw2,even,odd,ntasks;
};


int main(int argc, char* argv[]) {    
    int nw1 = 2;
    int nw2 = 3;
    int ntasks = 1000;
    if (argc>1) {
        if (argc<3) {
            printf("use: %s #w1 #w2 #ntasks\n", argv[0]);
            return -1;
        }
        nw1   = atoi(argv[1]);
        nw2   = atoi(argv[2]);
        ntasks= atoi(argv[3]);
    }
    
    ff_farm farm;   
    Emitter E(nw1, nw2, ntasks);
    farm.add_emitter(&E);
    std::vector<ff_node *> w;
    for(int i=0;i<nw1;++i) 
        w.push_back(new Worker1);
    for(int i=0;i<nw2;++i) 
        w.push_back(new Worker2);
    farm.add_workers(w); 
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }
    printf("done\n");
    return 0;
}
