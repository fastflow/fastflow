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
 * Very basic test for the FastFlow pipeline (1-stage pipeline).
 *
 */

#include <iostream>
#include <cstdio>
#include <ff/ff.hpp>

using namespace ff;

class Stage: public ff_node {
public:
    Stage():counter(0) {}

    int svc_init() {
        printf("Hello, I'm Stage\n");
        return 0;
    }
    void * svc(void *) {
        if (++counter > 10) return NULL;
        printf("Hi!\n");
        return GO_ON;
    }
    void  svc_end() {
        printf("Goodbye....\n");
    }
private:
    long counter;
};


int main() {
    ff_pipeline pipe;
    pipe.add_stage(new Stage);
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    std::cerr << "DONE, pipe  time= " << pipe.ffTime() << " (ms)\n";
    return 0;
}
