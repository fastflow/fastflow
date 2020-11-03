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
/* Testing: 
 * 
 *   Left -->|    
 *           |    | -> pipe(Right -> Right)
 *     ...   | -->|  
 *           |    | -> pipe(comb(Right,Right))
 *   Left -->|
 *                          
 * In this simple test, the right-hand side Workers are "heterogeneous", the top one is 
 * a pipeline (starting with a multi-input node), whereas the bottom one is a pipeline of a single combine node. 
 */
/*
 *
 * Author: Massimo Torquati
 */

#include <mutex>
#include <iostream>
#include <string>
#include <ff/ff.hpp>

using namespace ff;

struct Left: ff_monode_t<long> {
	Left(long ntasks): ntasks(ntasks) {}

	long* svc(long*) {
        for (long i=1;i<=ntasks;++i)
            ff_send_out(&i);        
		return EOS; 
	}
    long ntasks;
};
struct Right: ff_minode_t<long> {
    Right(const bool propagate=false):propagate(propagate) {}
	long* svc(long* in) {
        if (propagate) return in;
        return GO_ON;
	}
    const bool propagate;
};


int main(int argc, char* argv[]) {
    long ntasks   = 1000000;
    long nlefts   = 4;
    
    std::vector<ff_node*> W1;
    for(long i=0;i<nlefts;++i) {
        W1.push_back(new Left(ntasks/nlefts));
    }
    std::vector<ff_node*> W2;
    ff_pipeline pipe;
    pipe.add_stage(new Right(true), true);
    pipe.add_stage(new Right(false), true);
    W2.push_back(&pipe);
    ff_comb comb(new Right(true), new Right(false), true,true);
    ff_pipeline pipe1;
    pipe1.add_stage(&comb);
    W2.push_back(&pipe1);
  
    ff_a2a a2a;
    a2a.add_firstset(W1, 0, true);
    a2a.add_secondset(W2);
    
    if (a2a.run_and_wait_end()<0) {
        error("running a2a\n");
        return -1;
    }
    std::cout << "Done\n";
    return 0;
}



