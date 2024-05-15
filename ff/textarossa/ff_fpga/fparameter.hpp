#ifndef __FPARAMETER_HPP__
#define __FPARAMETER_HPP__

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/opencl.hpp>

#include <cstddef>


struct FParameter
{
    enum Type {
        H2D,    // Host to Device                       (IN)
        D2H,    // Device to Host                       (OUT)
        H2H,    // Host to Device and Device to Host    (INOUT)
        D2D,    // Device only
        VAL,    // Scalar value
        USR     // User data storage
    };

    enum TTL {
        SINGLE,         // Automatically freed after one use
        USER_MANAGED    // User must free after use
    };

    // Host
    void * const data;      // Pointer to data
    size_t size;            // Size of data in bytes

    // Description
    Type type;              // Type of buffer
    TTL ttl;                // Time to live

    // Device
    cl::Buffer buffer;      // OpenCL buffer

    bool set_argument_once;

    // General constructor
    FParameter(void * const data, size_t size, Type type, TTL ttl, bool set_argument_once = false)
    : data(data)
    , size(size)
    , type(type)
    , ttl((type == USR ? USER_MANAGED : ttl))
    , set_argument_once(set_argument_once)
    {}

    // Construct a buffer from host data and automatically free it after first use
    FParameter(void * const data, size_t size, Type type)
    : FParameter(data, size, type, SINGLE)
    {}

    // Construct a H2H buffer from host data and let the user free it
    FParameter(void * const data, size_t size)
    : FParameter(data, size, H2H, USER_MANAGED)
    {}

    // Construct a device only buffer with size in bytes and let the user free it
    FParameter(size_t size)
    : FParameter(nullptr, size, D2D, USER_MANAGED)
    {}

    cl_mem_flags getFlags()
    {
        if (type == D2D && data != nullptr) {
            std::cerr << "ERROR: data should be nullptr" << std::endl;
            exit(-1);
        }

        cl_mem_flags flags = 0;

        if (data != nullptr) {
            flags |= CL_MEM_USE_HOST_PTR;
        }

        switch (type) {
            case H2D:
                flags |= CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY;
                break;
            case D2H:
                flags |= CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY;
                break;
            case H2H:
                flags |= CL_MEM_READ_WRITE;
                break;
            case D2D:
                flags |= CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE;
                break;
            case VAL:
            case USR:
            default:
                break;
        }

        return flags;
    }

    void allocateBuffer(const FDevice & device)
    {
        if (type == VAL or type == USR) return;

        cl_int err;
        OCL_CHECK(err, buffer = cl::Buffer(device.context_, getFlags(), size, data, &err));
    }

    ~FParameter()
    {
        if (ttl == SINGLE) {
            free(data);
        }
    }
};

#endif // __FPARAMETER_HPP__