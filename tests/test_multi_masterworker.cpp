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
#include <ff/ff.hpp>
  

using namespace ff;

// generic worker
struct Worker: ff_node {
    Worker(int id):id(id) {}
    void * svc(void * task) {
        printf("FARM(%d) W(%ld): got %ld, now sleeping for a while....\n", id,get_my_id(), *(long*)task);
        usleep(10000);
        return task;
    }
    int id;

    void svc_end() {
        printf("FARM(%d) W(%ld) EXITING\n",id,get_my_id());
    }
};

class Emitter: public ff_node {
public:
    Emitter(int id, ff_loadbalancer *const lb):eosarrived(false),_id(id), numtasks(0),lb(lb) {}

    int svc_init(){
        printf("Emitter %d started\n", _id);
        return 0;
    }

    void * svc(void * task) {
        long * t = (long *)task;

        if (*t % 2)  {
            if (lb->get_channel_id()>=0) { // received from workers
                if (--numtasks == 0 && eosarrived)
                    return NULL;
            }
            return GO_ON;
        }
        ++numtasks;
        ++(*t);
        return task;
    }
    void eosnotify(ssize_t id) {
        if (id == -1) {
            eosarrived= true;
            if (numtasks==0) lb->broadcast_task(EOS);
        }
    }
protected:
    bool eosarrived;
    int _id;
    long numtasks;
    ff_loadbalancer *const lb;
};


class mainEmitter: public ff_node {
public:
    mainEmitter(int streamlen):streamlen(streamlen) {}

    void * svc(void *) {
        for(long i=1;i<=streamlen;++i)
            ff_send_out(new long(i));
        return NULL;
    }
private:
    int streamlen;
};



int main(int argc, char * argv[]) {
    int nworkers = 3;
    int streamlen= 1000;
    if (argc>1) {
        if (argc!=3) {
            std::cerr << "use: " 
                      << argv[0] 
                      << " nworkers streamlen\n";
            return -1;
        }
        
        nworkers=atoi(argv[1]);
        streamlen=atoi(argv[2]);
    }
    if (nworkers<=0 || streamlen<=0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }

    ff_farm mainFarm;
    mainEmitter mainE(streamlen);
    mainFarm.add_emitter(&mainE);
    mainFarm.set_scheduling_ondemand();

    std::vector<ff_node *> w;
    std::vector<ff_node *> w2;
    for(int i=0;i<nworkers;++i) {
        ff_farm *W = new ff_farm;
        Emitter *e = new Emitter(i,W->getlb());
        W->add_emitter(e);
        
        w.push_back(new Worker(i));
        w.push_back(new Worker(i));

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
