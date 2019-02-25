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
 * Date  : August 2014
 *         
 */
// simple test for the ff_taskf pattern
#include <ff/ff.hpp>
#include <ff/taskf.hpp>
#include <ff/parallel_for.hpp>
using namespace ff;

void F(long *E) {
    printf("Executing %ld\n", *E);
    delete E;
}

int main(int argc, char *argv[]) {
    int W = ff_numCores();
    if (argc>1) W = atoi(argv[1]);
    ff_taskf taskf(W);

    // start immediatly the scheduler and all worker threads
    taskf.run(); 

    taskf.AddTask(F, new long(1));
    taskf.AddTask(F, new long(2));
    taskf.AddTask(F, new long(3));

    taskf.wait(); // barrier

    // add another task, here the scheduler is stopped
    taskf.AddTask(F, new long(4)); 
    taskf.AddTask(F, new long(5));  

    // run the scheduler using 1 thread and then barrier
    taskf.run_then_freeze(1); 

    // here the scheduler is stopped
    taskf.AddTask(F, new long(6));
    taskf.AddTask(F, new long(7));
    taskf.AddTask(F, new long(8));

    // run the scheduler and then barrier
    taskf.run_then_freeze(2); 

    // here the scheduler is stopped
    taskf.AddTask(F, new long(9));  
    taskf.AddTask(F, new long(10));

    taskf.run_then_freeze();

    // now stressing the parallel for

    ParallelFor pf(W+4,true);
    pf.disableScheduler();
    std::atomic_long K;
    K.store(0);

    for(long i=1;i<=100;++i) {
	pf.parallel_for(0,10,1,1,[&K](const long j) { K+=j; }, 1+i%4);
	printf("."); fflush(stdout);
	pf.threadPause();
    }
    printf("\n");
    if (K != 4500) abort();
    printf("K=%ld\n", K.load());
    pf.threadPause();

    // re-start the taskf scheduler
    taskf.run();
    for(long i=1;i<=100;++i) 
	taskf.AddTask(F, new long(i));
    taskf.wait();

    return 0;
}
