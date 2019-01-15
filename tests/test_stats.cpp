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
 */
/* Author: Massimo Torquati
 *
 * Simple test showing how to get statistics at run-time. 
 */

#include <cstdlib>
#include <random>
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

struct Stage: ff_node_t<long> {
    Stage(int n):n(n), mt(n) { }
    
    long *svc(long *in) {
        
        switch(n) {
        case 0: {
            for(long i=0;i<2000;++i) 
                ff_send_out(new long(i));
            return EOS;
        } break;
        case 1: {
            delete in;
            in = new long(100);
            std::uniform_int_distribution<long> dist(1000, 10000);
            usleep(dist(mt));
            return in;
        } break;
        case 2: {
            delete in;
            std::uniform_int_distribution<long> dist(500, 5000);
            usleep(dist(mt));
            return GO_ON;
        } break;
        default:
            abort();
        }
        return EOS;
    }
    int n=-1;
    std::mt19937 mt;
};

int main() {
    
    Stage stage1(0);
    ff_Farm<long,long> farm([]() {
            std::vector<std::unique_ptr<ff_node> > W;
            W.push_back(make_unique<Stage>(1));
            W.push_back(make_unique<Stage>(1));
            return W;
        } ());
    Stage stage3(2);

    ff_Pipe<> pipe(stage1, farm, stage3);
    pipe.run();
    
    while(!pipe.done()) {
        sleep(1);
        pipe.ffStats(std::cout);
        std::cout<< "\n\n";
    }

    pipe.wait();
    std::cout << "DONE\n";
    return 0;
}
