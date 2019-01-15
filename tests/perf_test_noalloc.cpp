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


//#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <stdint.h>

#include <ff/mapping_utils.hpp>
#include <ff/ff.hpp>

using namespace ff;

/*
static inline unsigned long getCpuFreq() {
    FILE       *f;
    unsigned long long t;
    float       mhz;

    f = popen("cat /proc/cpuinfo |grep MHz |head -1|sed 's/^.*: //'", "r");
    fscanf(f, "%f", &mhz);
    t = (unsigned long)(mhz * 1000000);
    pclose(f);
    return (t);
}

*/
class Worker: public ff_node {
public:
    Worker(long long nticks):nticks(nticks) {};

    void * svc(void *) {
	ticks_wait(nticks);
        return GO_ON;
    }
private:
    long long nticks;
};

class Emitter: public ff_node {
public:
    Emitter(unsigned int numtasks):numtasks(numtasks) {};

    void * svc(void * ) {	
	for(unsigned int i=0;i<numtasks;++i)
	    ff_send_out(&i);
	
	return NULL;       
    }
private:
    unsigned int numtasks;
};

void usage(char * name) {
    std::cerr << "usage: \n";
    std::cerr << "      " << name << " num-buffer-entries streamlen nworkers nticks\n";
}

int main(int argc, char * argv[]) {
    unsigned int buffer_entries = 512;
    unsigned int numtasks       = 10000000;
    unsigned int nworkers       = 3;
    long long nticks            = 1000;
    if (argc>1) {
        if (argc!=5) {	
            usage(argv[0]);
            return -1;
        }
        
        buffer_entries = atoi(argv[1]);
        numtasks       = atoi(argv[2]); 
        nworkers       = atoi(argv[3]);
        nticks         = atoi(argv[4]);
    }

    ff_farm farm(false, nworkers*buffer_entries);    
    Emitter E(numtasks);
    farm.add_emitter(&E);
    std::vector<ff_node *> w;
    for(unsigned int i=0;i<nworkers;++i) w.push_back(new Worker(nticks));
    farm.add_workers(w);
    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }

    printf("Time: %g (ms)\n", farm.ffTime());

    printf("SEQ:\n");
    
    ff::ffTime(ff::START_TIME);
    for(unsigned int i=0;i<numtasks;++i) 
	ticks_wait(nticks);
    ff::ffTime(ff::STOP_TIME);
    printf("Time: %g (ms)\n", ff::ffTime(ff::GET_TIME));
    printf("Ticks =~ %f (usec)\n",(nticks / (1.0*(ff_getCpuFreq()/1000000.0))));



    return 0;
}
