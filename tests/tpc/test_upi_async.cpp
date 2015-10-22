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

#if !defined(FF_TPC)
// needed to enable the TPC FastFlow run-time
#define FF_TPC
#endif

#include <ff/tpcnode.hpp>
using namespace ff;

// kernel id inside the FPGA
#define KERNEL1_ID	  2
#define KERNEL2_ID    1
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
//
// void kernel2(uint32_t const *const d1sz, uint32_t const d1[MAX_SIZE],
//     uint32_t const *const d2sz, uint32_t d2[MAX_SIZE])
// {
//   uint32_t const s1 = *d1sz;
//   uint32_t const s2 = *d2sz;
//   for(uint32_t i = 0; i < s2; ++i)
//     #pragma HLS pipeline
//     d2[i] = (i < s1 ? d1[i] : 1);
// }

struct TaskCopy: public baseTPCTask<TaskCopy> {
    TaskCopy():in(nullptr),out(nullptr), sizein(0), sizeout(0) {}
              
    TaskCopy(uint32_t *in, uint32_t sizein, uint32_t *out, uint32_t sizeout):
        in(in),out(out),sizein(sizein),sizeout(sizeout) {}

    void setTask(const TaskCopy *t) { 

        setKernelId(KERNEL1_ID);

        setInPtr(&t->sizein, 1, 
                 BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::RELEASE);
        setInPtr(t->in, t->sizein, 
                 BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::RELEASE);
        setInPtr(&t->sizeout, 1, 
                 BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::RELEASE);
        // neither copied back nor released
        setOutPtr(t->out, t->sizeout, 
                  BitFlags::DONTCOPYBACK, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);
    }

    uint32_t *in,*out;
    uint32_t  sizein, sizeout;
};


struct Task: public baseTPCTask<Task> {
    Task():in(nullptr),sizein(0),start(0),stop(0),result(nullptr) {}
              
    Task(uint32_t *in, uint32_t sizein, uint32_t start, uint32_t stop, uint32_t *result):
        in(in),sizein(sizein),start(start),stop(stop),result(result) {}

    void setTask(const Task *t) { 

        setKernelId(KERNEL2_ID);

        setInPtr(&t->start, 1, 
                 BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);
        setInPtr(&t->stop,  1, 
                 BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);

        // not copied into the device, 
        // reusing the previous version (allocated by previous kernel)
        // the memory is not released at the end
        setInPtr(t->in, t->sizein, 
                 BitFlags::DONTCOPYTO, BitFlags::REUSE, BitFlags::DONTRELEASE);

        setOutPtr(t->result, 1, 
                  BitFlags::COPYBACK, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);

    }

    uint32_t *in;
    uint32_t  sizein;
    uint32_t  start, stop; 
    uint32_t *result;
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

static 
void check(uint32_t to, uint32_t from, uint32_t result) {
    if (result != ingauss(to, from+1))
        std::cerr << "Wrong result: " << result << " (expected: " 
                  << ingauss(to, from+1) << ")\n"; 
    else
        std::cout << "RESULT OK " << result << "\n";    
}

/* ----------------------------- */

// RePaRa code:
//   const size_t size = 256;
//   uint32_t waits[size];
//   uint32_t waits2[size] {0};
//   std::vector<uint32_t> results(4,0);
//
//   [[rpr::kernel, rpr::in(waits,size) rpr::out(waits2),
//     rpr::target(FPGA), rpr::keep(waits2) ]]
//   kernel2(size, waits, size, waits2);
//
//   [[rpr::kernel, rpr::async, 
//     rpr::in(waits2), rpr::out(results),
//     rpr::target(FPGA) ]]
//   kernel1(0, 64, waits2, &results[0]);
//
//   [[rpr::kernel, rpr::async, 
//     rpr::in(waits2), rpr::out(results),
//     rpr::target(FPGA) ]]
//   kernel1(64, 128, waits2, &results[1]);
//
//   [[rpr::kernel, rpr::async, 
//     rpr::in(waits2), rpr::out(results),
//     rpr::target(FPGA) ]]
//   kernel1(128, 192, waits2, &results[2]);
//
//   [[rpr::kernel, rpr::async,
//     rpr::in(waits2), rpr::out(results),
//     rpr::target(FPGA) ]]
//   kernel1(192, 255, waits2, &results[3]);
//
//   [[rpr::kernel, rpr::sync]];
//
//   check(0,   64, results[0]);
//   check(64, 128, results[1]);
//   check(128,192, results[2]);
//   check(192,255, results[3]);   
//      
//
int main() {
    const size_t size = 256;
    uint32_t waits[size];
    uint32_t waits2[size] {0};
    std::vector<uint32_t> results(4,0);

    for (size_t j = 0; j < size; ++j)
        waits[j] = j + 1;

    // device memory allocator shared between the two kernels
    ff_tpcallocator alloc;

    /* --- first kernel (kernel2) --- */
    TaskCopy k1(waits, size, waits2, size);
    ff_tpcNode_t<TaskCopy> copy(k1, &alloc);
    /* ------------------------------ */

    std::vector<Task>                tasks;
    std::vector<ff_tpcNode_t<Task> > nodes;

    for(size_t i=0; i<4; ++i) 
        tasks.push_back(Task(waits,size, i*64, i*64+64, &results[i]));
    for(size_t i=0;i<4; ++i)
        nodes.push_back(ff_tpcNode_t<Task>(tasks[i], &alloc));

    /* ------------------------------ */

    // running first kernel
    if (copy.run_and_wait_end()<0) {
        error("running first kernel\n");
        return -1;        
    }

    // running the four tpcnode instances asynchronously
    for(size_t i=0; i<4; ++i) {
        if (nodes[i].run()) {
            error("running nodes %d\n", i);
            return -1;
        }
    }

    for(size_t i=3; i>0; --i) {
        if (nodes[i].wait()<0) {
            error("waiting nodes %d\n", i);
            return -1;
        }
    }

    // checking the results
    check(0,   64, results[0]);
    check(64, 128, results[1]);
    check(128,192, results[2]);
    check(192,255, results[3]);
        
    return 0;
}
    
