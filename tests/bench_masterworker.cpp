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
 * Master-worker pattern repeated multiple times.
 *
 *
 *         ------------------------                  
 *        |              -----     |
 *        |        -----|     |    |
 *        v       |     |  W  |----
 *      -----     |      -----
 *     |  E  |----|        .
 *     |     |    |        .
 *      -----     |        .
 *        ^       |      -----
 *        |        -----|  W  | ---
 *        |             |     |    |
 *        |              -----     |
 *         ------------------------    
 *
 *
 */      

#include <vector>
#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

struct Worker: ff_node_t<long> {
  long *svc(long *in) {

      volatile size_t i=0;
      while(i<1000) {
          __asm__("nop");
          i = i +1;
      }
      
      return in;
  }
};

struct Emitter: ff_node_t<long> {
    Emitter(const size_t size): size(size) {}
    int svc_init() {
        received = 0;
        return 0;
    }
    long *svc(long *in) {
        if (in == nullptr) {
            for(size_t i=0;i<size;++i) ff_send_out((long*)(i+1));
            return GO_ON;
        }
        ++received;
        if (received >= size) return EOS;
        return GO_ON;
  }
  const size_t size=0;
    size_t received = 0;
};

int main(int argc, char *argv[]) {
    size_t nw = 4;
    size_t size = 10000;
    
    if (argc > 1) {
        if (argc != 3)  {
            std::cerr << "use: " << argv[0] << " workers size\n";
            return -1;
        }
        nw = std::stol(argv[1]);
        size = std::stol(argv[2]);        
    }
    
    Emitter E(size);
    
#if 0
    auto farm = make_MasterWorker<Worker>(make_unique<Emitter>(size), nw);
#else    
    ff_Farm<> farm([&]() {
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nw;++i) 
                W.push_back(make_unique<Worker>());
            return W;
        } (), E);
    farm.remove_collector();
    farm.wrap_around();
#endif

    for(int i=0;i<10;++i) {
        unsigned long start = ffTime(START_TIME);
        farm.run_then_freeze();
        farm.wait_freezing();
        unsigned long end   = ffTime(STOP_TIME);
        std::cout << "Time (ms): " << end-start << "\n";
    }
    return 0;
}
