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
/* Author: Massimo
 * Date  : January 2014
 *
 */
/*  
 *         
 *    Stage0 -----> Stage1 -----> Stage2 -----> Stage3
 *    ^  ^ ^             |          |              | 
 *     \  \ \------------           |              |
 *      \  \-------------------------              |
 *       \-----------------------------------------                                              
 *
 */

#include <ff/pipeline.hpp>
#include <ff/farm.hpp> 

using namespace ff;
long const int NUMTASKS=10;

struct Stage0: ff_minode {
    int svc_init() { counter=0; return 0;}
    void *svc(void *task) {
        if (task==NULL) {
            for(long i=1;i<=NUMTASKS;++i)
                ff_send_out((void*)i);
            return GO_ON;
        }
        printf("STAGE0 got back %ld from %zd\n", (long)task, get_channel_id());
        ++counter;
        if (counter == NUMTASKS) return NULL;
        return GO_ON;
    }
    long counter;
};

struct Stage1: ff_monode {
    void *svc(void *task) {
        if ((long)task & 0x1) {
            printf("Stage1 sending back=%ld\n", (long)task);
            ff_send_out_to(task, 0); // sends odd tasks back
        } else {
            ff_send_out_to(task, 1);
        }
        return GO_ON;
    }
    void eosnotify(ssize_t) {
        ff_send_out((void*)EOS);
    }
};
struct Stage2: ff_monode {
    void *svc(void *task) {
        if ((long)task <= (NUMTASKS/2)) {
            printf("Stage2 sending back=%ld\n", (long)task);
            ff_send_out_to(task, 0); // sends even tasks less than... back
        } else 
            ff_send_out_to(task, 1);
        return GO_ON;
    }
    void eosnotify(ssize_t) {
        ff_send_out((void*)EOS);
    }
};
struct Stage3: ff_node {
    void *svc(void *task) { 
        printf("Stage3 got task %ld\n", (long)task);
        assert(((long)task & ~0x1) && (long)task>(NUMTASKS/2));
        return task; 
    }
};

int main() {
    Stage0 s0; Stage1 s1; Stage2 s2; Stage3 s3;
#if 0
    ff_pipeline pipe;
    pipe.add_stage(&s0);
    pipe.add_stage(&s1);
    pipe.wrap_around(true);
    pipe.add_stage(&s2);
    pipe.wrap_around(true);
    pipe.add_stage(&s3);
    pipe.wrap_around();
#endif
    ff_pipeline pipe1;
    pipe1.add_stage(&s0);
    pipe1.add_stage(&s1);
    pipe1.wrap_around();

    ff_pipeline pipe2;   
    pipe2.add_stage(&pipe1);
    pipe2.add_stage(&s2);
    pipe2.wrap_around();
 
    ff_pipeline pipe;    
    pipe.add_stage(&pipe2);
    pipe.add_stage(&s3);

    pipe.wrap_around();

    pipe.run_and_wait_end();

    printf("DONE\n");
    return 0;
}
