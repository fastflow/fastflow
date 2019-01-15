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
 * This program tests ff_send_out function in a farm with 
 * a feedback channel.
 *
 */


#include <iostream>
#include <vector>
#include <ff/ff.hpp>
#include <ff/allocator.hpp>

using namespace ff;

static ff_allocator ffalloc;


class Worker: public ff_node {
public:
    void * svc(void * task) {
        int * t = (int *)task;
        std::cout << "Worker id= " << get_my_id() 
                  << " got task " << *t << "\n";

        if (*t>0) {
            *t = *t -1;
            task = t;
            ff_send_out(task);
            return GO_ON;
        } 
        *t = -1;
        task = t;
        return task;
    }
    void svc_end() {
        std::cout << "Worker id= " << get_my_id() << " received EOS\n";
    }

    
};

// the load-balancer filter
class Emitter: public ff_monode {
public:
    Emitter(int streamlen):streamlen(streamlen) {
        srandom(::getpid()+(getusec()%4999));
    };

    // called just one time at the very beginning
    int svc_init() {
        if (ffalloc.registerAllocator()<0) {
            error("Emitter, registerAllocator fails\n");
            return -1;
        }

        return 0;
    }

    void * svc(void * task) {
        int * t = (int *)task;

        if (!t) {
            // start generating the stream...
            for(int i=0;i<streamlen;++i) {
                t = (int *)ffalloc.malloc(sizeof(int));
                *t = i;
                ff_send_out(t);
            }
            task = GO_ON; // we want to keep going
        }
        return task;
    }

    void svc_end() {
        printf("Emitter received EOS\n");
        broadcast_task(EOS);
    }

    
private:
    int streamlen;
};


// the gatherer filter
class Collector: public ff_node {
public:
    Collector(int streamlen):streamlen(streamlen),cnt(0) {}
    int svc_init() {
        if (ffalloc.register4free()<0) {
            error("Collector, register4free fails\n");
            return -1;
        }
        return 0;
    }
    void * svc(void * task) {
        int * t = (int*)task;
        if (*t != -1) {
            std::cout << "Collector got task " << *t << " cnt= " << cnt << "\n";
            return task;
        }

        if (++cnt == streamlen) {
            std::cout << "Collector generating EOS\n";
            ffalloc.free(task);
            return EOS;
        }
        std::cout << "Collector got -1 cnt= " << cnt << "\n";
        return GO_ON;
    }
private:
    int streamlen;
    int cnt;
};



int main(int argc, char * argv[]) {
    int streamlen = 100;
    if (argc>1) {
        if (argc!=2) {
            std::cerr << "use: "  << argv[0] << " streamlen\n";
            return -1;
        }

        streamlen=atoi(argv[1]);
    }

    // init FastFlow allocator
    ffalloc.init();

    ff_farm farm;

    Emitter e(streamlen);
    Collector c(streamlen);
    farm.add_emitter(&e);
    farm.add_collector(&c);

    std::vector<ff_node *> w;
    w.push_back(new Worker);
    w.push_back(new Worker);
    w.push_back(new Worker);
    farm.add_workers(w);

    farm.wrap_around();

    if (farm.run_and_wait_end()<0) {
        error("running farm with feedback\n");
        return -1;
    }

	farm.ffStats(std::cerr);
	
    return 0;
}
