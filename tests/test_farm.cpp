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

#include <ff/farm.hpp>
  
using namespace ff;

// sequential function
static inline long Func(long r, long i) {
    usleep((r/i) % 1000);
    return r/i;
}

#if !defined(SEQUENTIAL)
struct task_t {
    task_t(long r, long i):r(r),i(i) {}
    long r, i;
};
static inline bool Wrapper(task_t &t) {
    t.r = Func(t.r, t.i);
    return true;
}
#endif

int main(int argc, char * argv[]) {

    if (argc<3) {
        printf("use: %s nworkers number\n",argv[0]);
        return -1;
    }
    int    nworkers=atoi(argv[1]);
    long   number  =atol(argv[2]);

    srandom(131071);
    
#if defined(SEQUENTIAL)
    long i=0;
    double k=0.0;
    do {
        k = Func(random(), i+1);
        if (k == number) { 
            printf("found %ld in i %ld iterations\n",number, i);
            break;
        }
    } while( ++i< (10*1e6) );
#else
    long i=0;
    task_t *r = NULL;
    FF_FARMA(farmA, Wrapper, task_t, nworkers, nworkers);
    FF_FARMARUN(farmA);
    do {
        FF_FARMAOFFLOAD(farmA, new task_t(random(),i+1));
        if (FF_FARMAGETRESULTNB(farmA, &r)) {
            if (r->r == number) {
                printf("found %ld in i %ld iterations\n",number, i);
                break;
            }
            delete r;
        }
    } while( ++i< (10*1e6) );
    FF_FARMAEND(farmA);
    if (r->r != number) {
        while(FF_FARMAGETRESULT(farmA,&r)) {
            if (r->r == number) {
                printf("found %ld after all iterations\n",number);
                break;
            }
            delete r;
        }
    }
    FF_FARMAWAIT(farmA);
#endif
    return 0;
}
