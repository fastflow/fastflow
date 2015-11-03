/* 
 * This file is also contained in the FastFlow directory  
 * <fastflow-dir>/tests/tpc.
 */
#include<iostream>
#if !defined(FF_TPC)
// needed to enable the TPC FastFlow run-time
#define FF_TPC
#endif
#include <ff/tpcnode.hpp>
using namespace ff;

// kernel id inside the device
#define KERNEL1_ID    2
#define KERNEL2_ID    1
#define MAX_SIZE    512

struct ExampleTaskCopy: public baseTPCTask<ExampleTaskCopy> {
    ExampleTaskCopy():in(nullptr),out(nullptr), sizein(0), sizeout(0) {}
              
    ExampleTaskCopy(uint32_t *in, uint32_t sizein, uint32_t *out, uint32_t sizeout):
        in(in),out(out),sizein(sizein),sizeout(sizeout) {}

    void setTask(const ExampleTaskCopy *t) { 

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


struct ExampleTask: public baseTPCTask<ExampleTask> {
    ExampleTask():in(nullptr),sizein(0),start(0),stop(0),result(nullptr) {}
              
    ExampleTask(uint32_t *in, uint32_t sizein, uint32_t start, uint32_t stop, uint32_t *result):
        in(in),sizein(sizein),start(start),stop(stop),result(result) {}

    void setTask(const ExampleTask *t) { 

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

        // copy back the result
        setOutPtr(t->result, 1, 
                  BitFlags::COPYBACK, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);
    }

    uint32_t *in;
    uint32_t  sizein;
    uint32_t  start, stop; 
    uint32_t *result;
};

/* ----------------------------- 
 * functions used for checking the result
 */
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

int main() {
    const size_t size = 256;

    // Each entry of the 'delays' vector contains the number of clock cycles 
    // to wait. This emulates a synthetic computation on each single element.
    uint32_t delays[size];
    uint32_t delays2[size] {0};
    std::vector<uint32_t> results(4,0);

    for (size_t j = 0; j < size; ++j)
        delays[j] = j + 1;

    // device memory allocator shared between the two kernels
    ff_tpcallocator alloc;

    /* ---. preparing the first kernel ---- */
    ExampleTaskCopy k1(delays, size, delays2, size);
    ff_tpcNode_t<ExampleTaskCopy> copy(k1, &alloc);
    /* ------------------------------------ */

    /* --- preparing the second kernel --- */
    std::vector<ExampleTask>                tasks(4);
    std::vector<ff_tpcNode_t<ExampleTask> > nodes(4);

    for(size_t i=0; i<4; ++i) 
        tasks.push_back(ExampleTask(delays,size, i*64, i*64+64, &results[i]));
    for(size_t i=0;i<4; ++i)
        nodes.push_back(ff_tpcNode_t<ExampleTask>(tasks[i], &alloc));
    /* ------------------------------------ */

    // running first kernel synchronously
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
    
    // waiting for the results
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
    
