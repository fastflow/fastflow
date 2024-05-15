#pragma once

#define UNUSED(x) (void)(x)

#define ALLOC_ALIGNMENT 4096

template <typename T>
T * aligned_alloc(const size_t n)
{
    void * ptr = nullptr;
    int ret = posix_memalign((void **)&ptr, ALLOC_ALIGNMENT, n * sizeof(T));
    if (ret != 0) {
        exit(ret);
    }
    return reinterpret_cast<T *>(ptr);
}

// XCL_MEM_DDR_BANK0
// XCL_MEM_DDR_BANK1
// XCL_MEM_DDR_BANK2
// XCL_MEM_DDR_BANK3

// #define MAX_HBM_PC_COUNT 32
// #define PC_NAME(n) (n | XCL_MEM_TOPOLOGY)
// const int pc_[MAX_HBM_PC_COUNT] = {
//     PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
//     PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15),
//     PC_NAME(16), PC_NAME(17), PC_NAME(18), PC_NAME(19), PC_NAME(20), PC_NAME(21), PC_NAME(22), PC_NAME(23),
//     PC_NAME(24), PC_NAME(25), PC_NAME(26), PC_NAME(27), PC_NAME(28), PC_NAME(29), PC_NAME(30), PC_NAME(31)};

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

auto get_time()
{
    return std::chrono::high_resolution_clock::now();
}

auto time_diff_ms(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}