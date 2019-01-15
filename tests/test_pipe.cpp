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
 * Simple 2-stage pipeline feeded by the main program (accelerator mode).
 *
 */

#include <ff/ff.hpp>
  
using namespace ff;

static inline long F(long i) { return i+1; }
static inline long G(long i) { return i*2; }

#if !defined(SEQUENTIAL)
struct fftask_t {
    fftask_t(long r):r(r) {}
    long r;
};
static inline fftask_t* wrapF(fftask_t* in, ff_node*const) { 
    long r = F(in->r);
    in->r=r;
    return in;
}
static inline fftask_t* wrapG(fftask_t *in, ff_node*const) { 
    long r = G(in->r);
    in->r=r;
    return in;
}
#endif

int main(int argc, char * argv[]) {
    long streamlen=1000;
    if (argc>1) {
        if (argc!=2) {
            printf("use: %s streamlen\n",argv[0]);
            return -1;
        }
        streamlen  =atol(argv[1]);
    }

#if defined(SEQUENTIAL)
    for(long i=0;i<streamlen;++i)
        printf("%ld ", G(F(i)));
#else
    ff_node_F<fftask_t> wrapf(wrapF), wrapg(wrapG);
    ff_Pipe<fftask_t, fftask_t> pipe(true, wrapf,wrapg); // accelerator mode turned on
    pipe.run_then_freeze();
    for(long i=0;i<streamlen;++i) {
        fftask_t *task = new fftask_t(i);
        pipe.offload(task);
    }
    pipe.offload(pipe.EOS);
    for(long i=0;i<streamlen;++i) {
        fftask_t *task = nullptr;
        pipe.load_result(task);
        assert(task != pipe.EOS);
        printf("result %ld\n", task->r);
        delete task;
    }
    pipe.wait();
#endif
    printf("\n");
    return 0;
}
