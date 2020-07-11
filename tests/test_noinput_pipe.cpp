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
 * Very basic test for the FastFlow pipeline with input throttling 
 * for the middle stage.
 *
 */

#include <cmath>
#include <iostream>
#include <ff/ff.hpp>

using namespace ff;

class Stage1: public ff_node {
public:
    Stage1(unsigned long streamlen):streamlen(streamlen)  {}

    void * svc(void *) {
        static unsigned long i=0;
        /*for(unsigned long i=1;i<=streamlen;++i)
            ff_send_out((void*)i);
        */
        if (++i>streamlen) return NULL;
        return (void*)i;
    }
private:
    unsigned long streamlen;
};

class Stage2: public ff_node {
public:
    Stage2():cnt(0){}

    void * svc(void * task) {
        if (task==NULL) {
            if (++cnt >= 10) {
                cnt =0;
                ff_node::input_active(true);
                return GO_ON;
            }
            printf("waiting cnt=%d\n", cnt);
            usleep(1000);
            return GO_ON;
        }
        ff_node::input_active(false);
        return task;
    }
private:
    int cnt;
};

class Stage3: public ff_node {
public:   
    void * svc(void *) {
        printf("got task\n");
        return GO_ON;
    }
};

int main(int argc, char * argv[]) {
    int streamlen = 1000;
    if (argc>1) {
        if (argc != 2) {
            printf("use: %s numtasks\n", argv[0]);
            return -1;
        }
        streamlen=atoi(argv[1]); 
    }
    ff_pipeline pipe(false, 10,10,true);
    pipe.add_stage(new Stage1(streamlen));
    pipe.add_stage(new Stage2());
    pipe.add_stage(new Stage3());
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    return 0;
}
