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
/* testing BLK and NBLK 
 *
 *  pipe(First, Farm(4), Last)
 */

/* Author: Massimo Torquati
 *
 */

#include <cstdio>
#include <ff/ff.hpp>
using namespace ff;

struct First: ff_node_t<long> {   
    long *svc(long *) {

        // enforces nonblocking mode since the beginning
        // regardless the compilation setting
        ff_send_out(NBLK); 

        for(size_t i=0;i<1000;++i) 
            ff_send_out((long*)(i+1));
        
        ff_send_out(BLK);

        for(size_t i=1000;i<2000;++i) 
            ff_send_out((long*)(i+1));

        ff_send_out(NBLK);

        for(size_t i=2000;i<3000;++i) 
            ff_send_out((long*)(i+1));
        
        ff_send_out(BLK);
        
        for(size_t i=3000;i<4000;++i) 
            ff_send_out((long*)(i+1));

        return EOS;
    }
};

struct Last: ff_node_t<long> {
    long *svc(long *task) {
        printf("Last received %ld\n", reinterpret_cast<long>(task));
        return GO_ON;
    }
};


struct Worker: ff_node_t<long> {
    long *svc(long *task) { 
        printf("Worker%ld, received %ld\n", get_my_id(), reinterpret_cast<long>(task));
        usleep(get_my_id()*1000);
        return task; 
    }
};



int main() {
    const size_t nworkers = 4;
    First first;
    Last  last;

    ff_Farm<long,long> farm(  [nworkers]() { 
	    std::vector<std::unique_ptr<ff_node> > W;
	    for(size_t i=0;i<nworkers;++i)  W.push_back(make_unique<Worker>());
	    return W;
	} () );
    
    farm.setInputQueueLength(nworkers*1, true);
    farm.setOutputQueueLength(nworkers*1, true);
    ff_Pipe<> pipe(first,farm,last);
    pipe.setXNodeInputQueueLength(1,true);
    pipe.setXNodeOutputQueueLength(1,true);
    pipe.run_then_freeze();
    pipe.wait();

    return 0;
}
