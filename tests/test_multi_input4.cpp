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
 *         
 *  Stage1 -----> Stage2 -----> Stage3
 *                 ^               |
 *                  \--------------         
 *
 */
#include <ff/ff.hpp>

using namespace ff;
long const int NUMTASKS=1000;

struct Stage1: ff_node {
    void *svc(void*) {
        for(long i=1;i<=NUMTASKS;++i)
            ff_send_out((void*)i);
        return NULL;
    }
};
struct Stage2: ff_minode {
    void *svc(void *task) {
        printf("Stage2 got task from %zd\n", get_channel_id());        
        if (fromInput()) return task; 
        return GO_ON;
    }
    void eosnotify(ssize_t) {
        ff_send_out((void*)EOS);
    }
};
struct Stage3: ff_node {
    void *svc(void *task) {
        long t=(long)task;
        printf("Stage3: got %ld, sending it back\n",t);
        return task;
    }
};
int main() {
    Stage1 s1; Stage2 s2; Stage3 s3;
#if 0    
    ff_pipeline pipe2;
    pipe2.add_stage(&s2);
    pipe2.add_stage(&s3);
    if (pipe2.wrap_around()<0) {
        error("wrap_around\n");
        return -1;
    }

    ff_pipeline pipe;
    pipe.add_stage(&s1);
    pipe.add_stage(&pipe2);
    pipe.run_and_wait_end();
#else
    ff_Pipe pipeI(s2,s3);
    pipeI.wrap_around();
    ff_Pipe pipe(s1,pipeI);
    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }
#endif    
    printf("DONE\n");
    return 0;
}
