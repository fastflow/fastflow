/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_UTILS_HPP_
#define _FF_UTILS_HPP_
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


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/time.h>
#include <cycle.h>
#include <iostream>
#include <pthread.h>

namespace ff {

enum { START_TIME=0, STOP_TIME=1, GET_TIME=2 };

static inline ticks ticks_wait(ticks t1) {
    ticks delta;
    ticks t0 = getticks();
    do { delta = getticks()-t0; } while (delta < t1);
    return delta-t1;
}

static inline void Barrier(int init = -1) {
    static int barrier = 0;
    static pthread_mutex_t bLock = PTHREAD_MUTEX_INITIALIZER;
    static pthread_cond_t  bCond = PTHREAD_COND_INITIALIZER;

    if (!barrier && init>0) { barrier = init; return; }

    pthread_mutex_lock(&bLock);
    if (!--barrier) pthread_cond_broadcast(&bCond);
    else {
        pthread_cond_wait(&bCond, &bLock);
        assert(barrier==0);
    }
    pthread_mutex_unlock(&bLock);
}

static inline void error(const char * str, ...) {
    va_list argp;

    //    fprintf(stderr, "%s, line %d: error: ", filename, lineno);
    va_start(argp, str);
    vfprintf(stderr, str, argp);
    va_end(argp);
}

// return current time in usec 
static inline unsigned long getusec() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (unsigned long)(tv.tv_sec*1e6+tv.tv_usec);
}

// compute a-b and return the difference in msec
static inline const double diffmsec(const struct timeval & a, 
                                    const struct timeval & b) {
    long sec  = (a.tv_sec  - b.tv_sec);
    long usec = (a.tv_usec - b.tv_usec);
    
    if(usec < 0) {
        --sec;
        usec += 1000000;
    }
    return ((double)(sec*1000)+ (double)usec/1000.0);
}


static inline double farmTime(int tag) {
    static struct timeval tv_start = {0,0};
    static struct timeval tv_stop  = {0,0};

    double res=0.0;
    switch(tag) {
    case START_TIME:{
        gettimeofday(&tv_start,NULL);
    } break;
    case STOP_TIME:{
        gettimeofday(&tv_stop,NULL);
        res = diffmsec(tv_stop,tv_start);
    } break;
    case GET_TIME: {
        res = diffmsec(tv_stop,tv_start);
    } break;
    default:
        res=0;
    }    
    return res;
}


} // namespace ff


#endif /* _FF_UTILS_HPP_ */
