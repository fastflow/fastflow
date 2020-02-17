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

/* Author: Massimo
 * Date  : May 2015
 *
 */                                    

/*
 *   pipe(farm, collector)
 *    
 *            ___________________
 *           |                   ^
 *           |                   |
 *           |      --> Worker --|--->
 *           v     |             |    |
 *       Emitter -- --> Worker --|---> -->Collector                  
 *           ^     |             v    |
 *           |      --> Worker --|--->
 *           |                   |
 *           |___________________v   
 *
 *
 */

#include <iostream>
#include <ff/ff.hpp>
using namespace ff;



class Emitter: public ff_node_t<long> {
public:
    Emitter(long ntasks):ntasks(ntasks) {}
    long* svc(long *t) {
	if (t == NULL) {
	    for(long i=0;i<ntasks;++i)
            ff_send_out((long*)(i+10));

        ntasks = ntasks / 2;
	    return GO_ON;
	}
	printf("Emitter %ld\n", (long)t);
	assert((long)t % 2);	
	if (--ntasks <= 0) {
        printf("sending EOS to workers\n");
        return EOS;

    }
	return (long*)((long)t +1);
    }
private:
    long      ntasks;
};

// multi-output worker 
struct Worker: ff_monode_t<long> {
    long* svc(long* task) {
        printf("Worker task=%ld\n", (long)task);
        if (((long)task %2) == 0) {
            printf("Worker task sending forward %ld\n", (long)task);
            ff_send_out_to(task, 1); // to the next stage 
        } else {
            printf("Worker sending back %ld\n", (long)task);
            ff_send_out_to(task, 0);  // channel 0 is the one that goes back to the emitter
        }
        return GO_ON;
    }
};

// multi-input stage
struct Collector: ff_minode_t<long> {
    long* svc(long* task) {
        printf("Collector received task = %ld\n", (long)(task));
        return GO_ON;
    }
};

int main(int argc, char* argv[]) {
    unsigned nworkers = 3;
    long     ntasks = 1000;
    if (argc>1) {
        if (argc < 3) {
            std::cerr << "use:\n" << " " << argv[0] << " numworkers ntasks\n";
            return -1;
        }
        nworkers  =atoi(argv[1]);
        ntasks    =atol(argv[2]);
        if (ntasks<2) {
            std::cerr << "ntasks must be greater than 2\n";
            return -1;
        }
    }   
    ff_Farm<long>   farm(  [&]() { 
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nworkers;++i)  {
		W.push_back(make_unique<Worker>());
	    }
	    return W;
	} () );
    Emitter E(ntasks);
    farm.remove_collector();
    farm.add_emitter(E);  
    farm.wrap_around();

    Collector C;
    ff_Pipe<> pipe(farm, C);
    pipe.run_and_wait_end();
    return 0;
}
