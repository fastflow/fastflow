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
 *
 */

#if !defined(FF_TASK_CALLBACK)
#define FF_TASK_CALLBACK
#endif

#include <vector>
#include <ff/ff.hpp>
#include <ff/taskf.hpp>

using namespace ff;

const int NTASKS = 2;

struct Worker1: ff_node_t<long> {

    void callbackIn(void *) {
        printf("Worker%ld callbackIn\n", get_my_id());
    }
    
    void callbackOut(void *) {
        printf("Worker%ld callbackOut\n", get_my_id());

    }
    long *svc(long *in) { 
        printf("Worker%ld got %ld\n", get_my_id(), *in);
        if (*in>0) --*in; 
        return in; 
    }
};

struct Worker2: ff_node_t<long> {

    void callbackIn(void *) {
        printf("Worker%ld callbackIn\n", get_my_id());
    }

    void callbackOut(void *) {
        printf("Worker%ld callbackOut\n", get_my_id());
    }
    long *svc(long *in) { 
        printf("Worker%ld got %ld\n", get_my_id(), *in);
        return in; 
    }
};


struct Emitter: ff_node_t<long> {

    Emitter(ff_loadbalancer *const lb):lb(lb) {}

    void callbackIn(void *p) {
        assert(reinterpret_cast<ff_loadbalancer*>(p) == lb);
        FF_IGNORE_UNUSED(p);
        printf("Emitter callbackIn\n");
    }
    
    void callbackOut(void *p) {
        assert(reinterpret_cast<ff_loadbalancer*>(p) == lb);
        FF_IGNORE_UNUSED(p);
        printf("Emitter callbackOut\n");
    }
    
    long *svc(long *in) {
        if (in == nullptr) {
            for(size_t i=0;i<NTASKS; ++i)
                lb->ff_send_out_to(new long(i), i % 2);
            return GO_ON;
        }
        
        const long &task = *in;
        if (task == 0) {
            ++zeros;
            if (zeros == NTASKS) return EOS;
        } else {
            lb->ff_send_out_to(in, next);
            next  = (next + 1) % 2;
        }
        return GO_ON;
    }

    
    int zeros = 0;
    int next  = 0;
    ff_loadbalancer *lb;
};


int main() {

    std::vector<std::unique_ptr<ff_node> > W;
    W.push_back(make_unique<Worker1>());
    W.push_back(make_unique<Worker2>());
    
    ff_Farm<> farm(std::move(W));
    Emitter E(farm.getlb());
    farm.add_emitter(E);
    farm.remove_collector();
    farm.wrap_around();

    farm.run_and_wait_end();
    
    return 0;
}
