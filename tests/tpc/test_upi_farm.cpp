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
/* Author: Massimo Torquati
 *         torquati@di.unipi.it
 *
 * Date:   September 2015
 *
 */
#include<iostream>
#include<ff/farm.hpp>

#if !defined(FF_TPC)
// needed to enable the TPC FastFlow run-time
#define FF_TPC
#endif

#include <ff/tpcnode.hpp>
using namespace ff;

// kernel id inside the FPGA
#define KERNEL1_ID	  1
#define MAX_SIZE 	512

// the kernel code is:
// void kernel1(uint32_t const *const idx_start, uint32_t const *const idx_stop,
//     uint32_t const cycles[MAX_SIZE], uint32_t *retval)
// {
//   uint32_t const w_start = *idx_start;
//   uint32_t const w_stop = *idx_stop;
//   uint32_t r = 0;
//   for (uint32_t i = w_start; i <= w_stop; ++i) {
//     #ifdef __SYNTHESIS__
//     {
//       #pragma HLS PROTOCOL fixed
//       uint32_t x = cycles[i];
//       wait(x);
//     }
//     #else
//     usleep(cycles[i]);
//     #endif
//     r += cycles[i];
//   }
//   *retval = r;
// }


struct Task: public baseTPCTask<Task> {
    Task():in(nullptr),sizein(0),start(0),stop(0),result(0) {}
              
    Task(uint32_t *in, uint32_t sizein, uint32_t start, uint32_t stop):
        in(in),sizein(sizein),start(start),stop(stop),result(0) {}

    void setTask(const Task *t) { 

        setKernelId(KERNEL1_ID);
        
        setInPtr(&t->start, 1, 
                 BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);
        setInPtr(&t->stop,  1,
                 BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);

        // the input array is copied only the first task
        setInPtr(t->in, t->sizein, 
                 first_time_flag?BitFlags::COPYTO:BitFlags::DONTCOPYTO, 
                 !first_time_flag?BitFlags::REUSE:BitFlags::DONTREUSE, 
                 BitFlags::DONTRELEASE);

        setOutPtr(&t->result, 1, 
                  BitFlags::COPYBACK, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);

        first_time_flag = false;
    }

    uint32_t *in;
    uint32_t  sizein;
    uint32_t  start, stop; 
    uint32_t  result;

    bool first_time_flag = true;
};

/* ----------------------------- */
// functions used for checking the result
static inline
uint32_t gauss(uint32_t const to) {
  return (to * to + to) / 2;
}

static inline
uint32_t ingauss(uint32_t const from, uint32_t to) {
  return gauss(to) - gauss(from);
}
/* ----------------------------- */

int main() {
    const size_t size = 256;
    uint32_t waits[size];
    for (int j = 0; j < size; ++j)
        waits[j] = j + 1;

    // task-farm scheduler (Emitter)
    struct Scheduler: ff_node_t<Task> {        
        Scheduler(uint32_t *waits, size_t size):waits(waits),size(size) {}
        Task *svc(Task *) {
            for(int i=10;i<200;++i)
                ff_send_out(new Task(waits, size, i, i+50));
            return EOS;                
        }
        uint32_t *waits;
        size_t    size;
    } sched(waits, size);

    // task-farm Collector
    struct Checker: ff_node_t<Task> {
        Task *svc(Task *in) {
            if (in->result != ingauss(in->start, in->stop+1))
                std::cerr << "Wrong result: " << in->result << " (expected: " 
                          << ingauss(in->start, in->stop+1) << ")\n"; 
            else
                std::cout << "RESULT OK " << in->result << "\n";
            return GO_ON;
        }
    } checker;

    ff_tpcallocator alloc;

    // this is the farm instance having 4 replicas of the tpcnode
    // the emitter of the farm is the scheduler (producing the stream)
    // the collector receives and check the results
    ff_Farm<> farm([&]() {
            const size_t nworkers = 4;
            std::vector<std::unique_ptr<ff_node> > W;
            for(size_t i=0;i<nworkers;++i)
                W.push_back(make_unique<ff_tpcNode_t<Task> >(&alloc));
            return W;
        } (), sched, checker);

    if (farm.run_and_wait_end()<0) {
        error("running farm\n");
        return -1;
    }

    return 0;
}
    
