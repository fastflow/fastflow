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
 * Basic tests for the combine_with_firststage/laststage helping functions.
 */

/* Author: Massimo Torquati
 *
 */

#include <cstdio>
#include <ff/ff.hpp>
using namespace ff;

const long FIRST=0;
const long LAST=10;

struct Stage:ff_node_t<long> {
    Stage(long n):n(n) {}
    long* svc(long* in) {
        if (n<0) return in;
        if (n==0) {
            printf("%ld: producing %ld\n",n,n+1);
            printf("%ld: producing %ld\n",n,n+2);
            ff_send_out((long*)(n+1));
            ff_send_out((long*)(n+2));
            return EOS;
        }
        if (n != LAST) {
            printf("%ld: producing %ld\n", n, (long)in +1 );
            ff_send_out((long*)((long)in+1));
        }
        return GO_ON;
    }
    long n=-1;
};

int main() {
    Stage first(FIRST), _1(1), _2(2), _3(3), _4(4), last(LAST);
    Stage *_5=new Stage(5);

    Stage *E = new Stage(-1);
    Stage *C = new Stage(-2);
    std::vector<ff_node*> W;
    W.push_back(new Stage(-3));
    W.push_back(new Stage(-4));
    ff_farm farm(W,*E,*C);
    farm.cleanup_all();
    farm.set_ordered(); // ordered farm
    
    ff_Pipe<> pipe0(farm);
    ff_Pipe<> pipe1(_2, pipe0);
    ff_Pipe<> pipe2(pipe1);
    ff_Pipe<> pipe3(pipe2);
    ff_Pipe<> pipeMain(pipe3);

    if (combine_with_firststage(pipe0, &_3)<0) return -1;
    if (combine_with_laststage(pipe0,  &_4)<0) return -1;
    if (combine_with_firststage(pipe2, &_1)<0) return -1;
    if (combine_with_firststage(pipe3, &first)<0) return -1;
    if (combine_with_laststage(pipeMain, _5, true)<0) return -1;
    if (combine_with_laststage(pipeMain, &last)<0) return -1;
    if (pipeMain.run_and_wait_end()<0) {
        error("running pipeMain\n");
        return -1;
    }

    return 0;
}
