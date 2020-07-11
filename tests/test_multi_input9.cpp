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
/*                       -----------------------
 *                      |                       |
 *                      v                       |   
 *                     --->  Stage1 ---> Stage2 --->             
 *                    |                             |
 *  Generator --> E-->|             ...             |---> C --> Gatherer
 *                    |                             |
 *                     --->  Stage1 ---> Stage2 --->
 *                      ^                       | 
 *                      |                       |  
 *                       -----------------------   
 *               
 *                      |<------ pipelines ------>|
 *
 * Stage1 is a multi-input node.
 * Stage2 is a multi-output node.
 *
 */
/*
 * Author: Massimo Torquati
 *
 */

#include <ff/ff.hpp>

using namespace ff;

struct Generator: ff_node_t<long> {
    const long ntasks=10;
    long* svc(long*) {
        for(long i=1;i<=ntasks;++i) {
            ff_send_out((long*)i);
        }
        return EOS;
    }
};
struct Gatherer: ff_node_t<long> {
    long* svc(long* in) {
        printf("Gatherer: received %ld\n", (long)in);
        return GO_ON;
    }
};

struct PipeWorker: ff_pipeline {
    struct Stage1: ff_minode_t<long> {
        int svc_init() {
            return 0;
        }
        long* svc(long* in) {
            if (fromInput()) {
                ++ntasks;
                return in;
            }
            --ntasks;
            if (ntasks==0 && eosreceived) return EOS;
            long tmp = (long)in;
            if (tmp & 0x1) return GO_ON;
            tmp++;
            ++ntasks;
            return (long*)tmp;
        }
        
        void eosnotify(ssize_t) {
            eosreceived=true;
            if (ntasks==0) ff_send_out(EOS);
        }
        long ntasks=0;
        bool eosreceived=false;
    };
    struct Stage2: ff_monode_t<long> {
        long* svc(long* in) {
            long tmp = (long)in;
            printf("Stage2: %ld\n", tmp);
            if (tmp & 0x1) ff_send_out_to(in, 1); // forward
            ff_send_out_to(in, 0); // backward
            return GO_ON;
        }    
    };
        
    PipeWorker() {
        Stage1* first   = new Stage1;
        Stage2* second  = new Stage2;
        add_stage(first,true);
        add_stage(second,true);
        wrap_around();
    }
    void *svc(void*) { abort();}
};


int main(int argc, char* argv[]) {
    const size_t nworkers=(argc>1)?std::stol(argv[1]):3;
    ff_Farm<long>   farm(  [nworkers]() { 
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nworkers;++i)  {
                W.push_back(make_unique<PipeWorker>());
            }
            return W;
        } () );

    Generator gen;
    Gatherer gat;
    ff_Pipe<> pipe(gen, farm, gat);

    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }

    return 0;
}
