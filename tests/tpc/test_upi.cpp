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
              
    Task(uint32_t *in, uint32_t sizein, uint32_t start, uint32_t stop, uint32_t *result):
        in(in),sizein(sizein),start(start),stop(stop),result(result) {}

    void setTask(const Task *t) { 

        setKernelId(KERNEL1_ID);
        setInPtr(&t->start, 1);
        setInPtr(&t->stop,  1);
        setInPtr(t->in, t->sizein);
        setOutPtr(t->result, 1);
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
/* ----------------------------- */

int main(int argc, char * argv[]) {
    const size_t size = 256;
    uint32_t waits[size];
    for (int j = 0; j < size; ++j)
        waits[j] = j + 1;
    
    uint32_t result = 0;
    
    Task tpct(waits, size, 45, 69, &result);
    ff_tpcNode_t<Task> tpcmap(tpct);

    if (tpcmap.run_and_wait_end()<0) {
        error("running tpcmap\n");
        return -1;
    }

    // checking results
    if (result != ingauss(45, 70))
        std::cerr << "Wrong return value: " << result << " (expected: " << ingauss(45, 70) << ")\n";
    std::cout << "result = " << result << std::endl;
    
    return 0;
}
    
