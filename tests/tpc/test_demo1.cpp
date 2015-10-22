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
 * Date:   October 2015
 *
 */
#include<iostream>

#if !defined(FF_TPC)
// needed to enable the TPC FastFlow run-time
#define FF_TPC
#endif
#include <ff/taskf.hpp>
#include <ff/tpcnode.hpp>
using namespace ff;

// kernel id inside the FPGA
#define KERNEL_ID	  1
#define MAX_LEN    1000 	// max array length
typedef int32_t elem_t;

struct Task: public baseTPCTask<Task,Task> {
    Task():a(nullptr),b(nullptr),c(nullptr),len(0),r(nullptr) {}
              
    Task(int32_t *r, uint32_t len, elem_t const *a, elem_t const *b, elem_t *c):
        a(a),b(b),c(c),len(len),r(r) {}

    void setTask(const Task *t) { 
        // sets the kernel id to launch
        setKernelId(KERNEL_ID);

        // first input param passed by-value
        setInVal(&t->len);  
        // second input param passed by reference
        setInPtr(t->a, t->len);
        // second input param passed by reference
        setInPtr(t->b, t->len);

        // first output param
        setOutPtr(t->c, t->len);

        // kernel function return value
        setReturnVar(t->r);
    }
    const elem_t  *a, *b; // input
    elem_t        *c;     // output
    uint32_t       len;   // input (by value)
    int32_t       *r;     // return val
};

/* -------- utility functions -------- */
/** Initializes an array to contain consecutive numbers starting at 1. **/
static inline void init_consec(uint32_t len, elem_t p[]) {
	for (uint32_t n = 1; n <= len; ++n, ++p) *p = n;
};
/** Checks the plausibility of an arbitrary length return vector. **/
static inline bool check(uint32_t len, elem_t *p) {
  bool ret = true;
  for (uint32_t n = 0; n < len; ++n) ret = ret && p[n] == (elem_t)(n+1) * 2;
  return ret;
}
static void check1(uint32_t r, elem_t& c) {
    bool const res1 = r == -6 && c == -1;
    fprintf(res1 ? stdout : stderr, "TEST 1 %s\n", res1 ? " PASSED" : " FAILED");
}
static void check2(uint32_t r, elem_t C[3]) {
    bool const res2 = r == 32 && C[0] == 5 && C[1] == 7 && C[2] == 9;
    fprintf(res2 ? stdout : stderr, "TEST 2 %s\n", res2 ? " PASSED" : " FAILED");
}
static void check3(uint32_t r, elem_t F[20]) {
    bool const res3 = r == 2870 && check(20, F);
    fprintf(res3 ? stdout : stderr, "TEST 3 %s\n", res3 ? " PASSED" : " FAILED");
}
static void check4(uint32_t r, std::vector<elem_t> &H) {
    bool const res4 = check(MAX_LEN, &H[0]);  // return value overflows, so is ignored
    fprintf(res4 ? stdout : stderr, "TEST 4 %s\n", res4 ? " PASSED" : " FAILED");
}
/* ----------------------------------- */

//  RePaRa code:
//
//  elem_t a = 2, b = -3, c = 0, r = 0;
//  [[ rpr::kernel, rpr::in(a,b), rpr::out(c,r), rpr::target(FPGA) ]]
//  vectoradddot(&r, 1, &a, &b, &c);
//  check1(r, c); 
// 
//  elem_t A[3] = { 1, 2, 3 }, B[3] = { 4, 5, 6 }, C[3] = { 0, 0, 0 };
//  [[ rpr::kernel, rpr::in(A,B), rpr::out(C,r), rpr::target(FPGA) ]]
//  vectoradddot(&r, 3, A, B, C);
//  check2(r, C);
//
//  elem_t E[20] = { 0 }, F[20] = { 0 };
//  [[ rpr::kernel, rpr::in(E), rpr::out(F,r), rpr::target(FPGA) ]]
//  vectoradddot(&r, 20, E, E, F);
//  check3(r, F);
//
//  std::vector<elem_t> G(MAX_LEN, 0), H(MAX_LEN, 0);
//  init_consec(MAX_LEN, &G[0]);
//  [[ rpr::kernel, rpr::in(G), rpr::out(H,r), rpr::target(FPGA) ]]
//  vectoradddot(&r, MAX_LEN, &G[0],&G[0],&H[0]);
//  check4(r, H);
//
int main() {
    auto task1 = []() {
        elem_t a = 2, b = -3, c = 0, r = 0;
        Task t(&r, 1, &a, &b, &c);        
        ff_tpcNode_t<Task> tpcf(t);
        tpcf.run_and_wait_end();
        check1(r,c);
    };
    auto task2 = []() {
        elem_t A[3] = { 1, 2, 3 }, B[3] = { 4, 5, 6 }, C[3] = { 0, 0, 0 }, r = 0; 
        Task t(&r, 3, A, B, C);
        ff_tpcNode_t<Task> tpcf(t);
        tpcf.run_and_wait_end();
        check2(r,C);
    };
    auto task3 = [&]() {
        elem_t E[20] = { 0 }, F[20] = { 0 }, r = 0;
        init_consec(20, E); 
        Task t(&r, 20, E, E, F);
        ff_tpcNode_t<Task> tpcf(t);
        tpcf.run_and_wait_end();
        check3(r,F);
    };
    auto task4 = [&]() {
        std::vector<elem_t> G(MAX_LEN, 0);
        std::vector<elem_t> H(MAX_LEN, 0);
        elem_t r;
        init_consec(MAX_LEN, &G[0]);
        Task t(&r, MAX_LEN, &G[0], &G[0], &H[0]);
        ff_tpcNode_t<Task> tpcf(t);
        tpcf.run_and_wait_end();
        check4(r,H);
    };
    const int async_degree = 4;
    ff_taskf taskf(async_degree);
    if (taskf.run()<0) {
        error("running taskf\n");
        return -1;
    }
    taskf.AddTask(task1);
    taskf.AddTask(task2);
    taskf.AddTask(task3);
    taskf.AddTask(task4);
    taskf.wait();
    return 0;
}
    

