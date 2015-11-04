#include<iostream>

#if !defined(FF_TPC)
// needed to enable the TPC FastFlow run-time
#define FF_TPC
#endif
#include <ff/taskf.hpp>
#include <ff/tpcnode.hpp>
using namespace ff;

// kernel id inside the FPGA
#define KERNEL_ID     1
#define MAX_LEN    1000 	// max array length
typedef int32_t elem_t;

struct VectorAddDotTask: public baseTPCTask<VectorAddDotTask,VectorAddDotTask> {
    VectorAddDotTask():a(nullptr),b(nullptr),c(nullptr),len(0),r(nullptr) {}
              
    VectorAddDotTask(int32_t *r, uint32_t len, elem_t const *a, elem_t const *b, elem_t *c):
        a(a),b(b),c(c),len(len),r(r) {}

    void setTask(const VectorAddDotTask *t) { 
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
static void check1(int32_t r, elem_t& c) {
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

int main() {
    auto task1 = []() {
        elem_t a = 2, b = -3, c = 0, r = 0;
        VectorAddDotTask t(&r, 1, &a, &b, &c);        
        ff_tpcNode_t<VectorAddDotTask> tpcf(t);
        tpcf.run_and_wait_end();
        check1(r,c);
    };
    auto task2 = []() {
        elem_t A[3] = { 1, 2, 3 }, B[3] = { 4, 5, 6 }, C[3] = { 0, 0, 0 }, r = 0; 
        VectorAddDotTask t(&r, 3, A, B, C);
        ff_tpcNode_t<VectorAddDotTask> tpcf(t);
        tpcf.run_and_wait_end();
        check2(r,C);
    };
    auto task3 = [&]() {
        elem_t E[20] = { 0 }, F[20] = { 0 }, r = 0;
        init_consec(20, E); 
        VectorAddDotTask t(&r, 20, E, E, F);
        ff_tpcNode_t<VectorAddDotTask> tpcf(t);
        tpcf.run_and_wait_end();
        check3(r,F);
    };
    auto task4 = [&]() {
        std::vector<elem_t> G(MAX_LEN, 0);
        std::vector<elem_t> H(MAX_LEN, 0);
        elem_t r;
        init_consec(MAX_LEN, &G[0]);
        VectorAddDotTask t(&r, MAX_LEN, &G[0], &G[0], &H[0]);
        ff_tpcNode_t<VectorAddDotTask> tpcf(t);
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

