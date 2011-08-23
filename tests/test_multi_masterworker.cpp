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

/* Tests a farm nesting case in which farm's workers are farm with feedback 
 * channel ( i.e. farm(D&C)):                              
 *                          ------>D&C
 *     (mainFarm)          |
 *         mainEmitter ----
 *                         |
 *                          ------>D&C    
 *
 *  where  D&C is:
 *
 *            -------------------------
 *           |                         |
 *           |        -------> Worker -
 *           v       |
 *        Emitter ---
 *           ^       | 
 *           |        -------> Worker -                
 *           |                         |
 *            -------------------------
 */



#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/node.hpp>
  

using namespace ff;

// generic worker
class Worker: public ff_node {
public:
    void * svc(void * task) {
        std::cerr << "sleeping....\n";
        usleep(10000);
        return task;
    }
};

class Emitter: public ff_node {
public:
    void * svc(void * task) {
        int * t = (int *)task;

        if (*t % 2)  return GO_ON;
        ++(*t);
        return task;
    }
};


class mainEmitter: public ff_node {
public:
    mainEmitter(int streamlen):streamlen(streamlen) {}

    void * svc(void *) {
        for(int i=1;i<=streamlen;++i)
            ff_send_out(new int(i));
        return NULL;
    }
private:
    int streamlen;
};



int main(int argc, 
         char * argv[]) {

    if (argc!=3) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " nworkers streamlen\n";
        return -1;
    }
    
    int nworkers=atoi(argv[1]);
    int streamlen=atoi(argv[2]);

    if (nworkers<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    ff_farm<> mainFarm;
    mainEmitter mainE(streamlen);
    mainFarm.add_emitter(&mainE);

    std::vector<ff_node *> w;
    std::vector<ff_node *> w2;
    for(int i=0;i<nworkers;++i) {
        Emitter * e = new Emitter;
        ff_farm<> * W = new ff_farm<>;
        W->add_emitter(e);
        
        w.push_back(new Worker);
        //w.push_back(new Worker);
        W->add_workers(w);
        w.clear();
        W->wrap_around();

        w2.push_back(W);
    }
    mainFarm.add_workers(w2);

    mainFarm.run();
    mainFarm.wait();

    std::cout<< "DONE\n";

    return 0;
}
