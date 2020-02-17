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
 *    first _    _ second
 *           |  |
 *    first _| -|- second
 *           |  |
 *    first -    - second
 *
 */ 
    
/* Author: Massimo Torquati
 *
 */
    
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

struct firstStage: ff_node_t<long> { 
    int svc_init() {
        std::cout << "firstStage started (" << get_my_id() << ")\n";
        return 0;
    }
    void svc_end() {
        std::cout << "firstStage ending (" << get_my_id() << ")\n";
    }
    long *svc(long*) {
        for(long i=0;i<10000;++i) {
            ff_send_out(new long(i));
        }	
        return EOS; // End-Of-Stream
    }
};
struct secondStage: ff_node_t<long> {  
    long *svc(long *task) {
        std::cout << "secondStage received " << *task << "\n";
        delete task;
        return GO_ON; 
    }
    void eosnotify(ssize_t) {
        std::cout << "EOS received by secondStage (" << get_my_id() << ")\n";
    }

}; 
int main() {
    firstStage    _1_1, _1_2, _1_3;
    secondStage   _2_1, _2_2, _2_3;
    
    std::vector<ff_node*> W1;
    W1.push_back(&_1_1);
    W1.push_back(&_1_2);
    W1.push_back(&_1_3);
    std::vector<ff_node*> W2;
    W2.push_back(&_2_1);
    W2.push_back(&_2_2);
    W2.push_back(&_2_3);
    
    ff_a2a a2a;
    
    a2a.add_firstset(W1);
    a2a.add_secondset(W2);
    
    if (a2a.run_and_wait_end()<0) {
        error("running A2A");
        return -1;
    }
    
    return 0;
}
