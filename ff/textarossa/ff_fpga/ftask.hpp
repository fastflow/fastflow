#pragma once

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/opencl.hpp>

#include "utils.hpp"
#include "fdevice.hpp"
#include "fparameter.hpp"

#include <iostream>
#include <string>
#include <vector>


using parameters_t = std::vector<FParameter *>;
using memories_t = std::vector<cl::Memory>;
using buffers_t = std::vector<cl::Buffer>;
using events_t = std::vector<cl::Event>;
using banks_t = std::vector<int>;

class FTask
{
private:

    cl_int err;

    parameters_t parameters;
    memories_t h2d_buffers;
    memories_t d2h_buffers;
    memories_t d2d_buffers;

    events_t write_events;
    events_t kernel_events;
    events_t read_events;

    //*************************************************************************
    //
    // Task Executor private functions
    //
    //*************************************************************************

    void prepare_buffers(FDevice & device)
    {
        for (FParameter * p : parameters) {
            p->allocateBuffer(device);
            switch (p->type) {
                case FParameter::Type::H2D:
                    h2d_buffers.push_back(p->buffer);
                    break;
                case FParameter::Type::D2H:
                    d2h_buffers.push_back(p->buffer);
                    break;
                case FParameter::Type::H2H:
                    h2d_buffers.push_back(p->buffer);
                    d2h_buffers.push_back(p->buffer);
                    break;
                case FParameter::Type::D2D:
                    d2d_buffers.push_back(p->buffer);
                    break;
                case FParameter::Type::VAL:
                case FParameter::Type::USR:
                default:
                    break;
            }
        }
    }

    void migrate_inputs(cl::CommandQueue & queue)
    {
        write_events.emplace_back();
        OCL_CHECK(err, err = queue.enqueueMigrateMemObjects(h2d_buffers, 0, nullptr, &write_events[0]));

        // TODO: check if CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED is needed or causes problems
        write_events.emplace_back();
        OCL_CHECK(err, err = queue.enqueueMigrateMemObjects(d2h_buffers, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &write_events[1]));
    }

    void setKernelArguments(cl::Kernel & kernel)
    {
        size_t argi = 0;
        for (FParameter * p : parameters) {
            switch (p->type) {
                case FParameter::Type::H2D:
                case FParameter::Type::D2H:
                case FParameter::Type::H2H:
                    OCL_CHECK(err, err = kernel.setArg(argi, p->buffer));
                    break;
                case FParameter::Type::D2D:
                    if (p->set_argument_once) {
                        OCL_CHECK(err, err = kernel.setArg(argi, p->buffer));
                        p->set_argument_once = false;
                    }
                    break;
                case FParameter::Type::VAL:
                    OCL_CHECK(err, err = kernel.setArg(argi, p->size, p->data));
                    break;
                case FParameter::Type::USR:
                default:
                    break;
            }
            argi++;
        }
    }

    void enqueueTask(cl::Kernel & kernel, cl::CommandQueue & queue)
    {
        kernel_events.emplace_back();
        OCL_CHECK(err, err = queue.enqueueTask(kernel, &write_events, &kernel_events[0]));
    }

    void migrate_outputs(cl::CommandQueue & queue)
    {
        read_events.emplace_back();
        OCL_CHECK(err, err = queue.enqueueMigrateMemObjects(d2h_buffers, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events, &read_events[0]));
    }


public:

    FTask()
    : parameters()
    , h2d_buffers()
    , d2h_buffers()
    , d2d_buffers()
    {}

    void setParameter(
        size_t idx,
        void * const data,
        size_t size,
        FParameter::Type type,
        FParameter::TTL ttl = FParameter::TTL::SINGLE
    )
    {
        if (idx >= parameters.size()) {
            parameters.resize(idx + 1);
        }
        parameters[idx] = new FParameter(data, size, type, ttl);
    }

    void setParameter(
        size_t idx,
        size_t size
    )
    {
        if (idx >= parameters.size()) {
            parameters.resize(idx + 1);
        }
        parameters[idx] = new FParameter(size);
    }

    void setParameter(
        size_t idx,
        FParameter * parameter
    )
    {
        if (idx >= parameters.size()) {
            parameters.resize(idx + 1);
        }
        parameters[idx] = parameter;
    }

    FParameter * getParameter(size_t idx)
    {
        if (idx >= parameters.size()) {
            std::cerr << "ERROR: parameter index out of bounds" << std::endl;
            exit(-1);
        }
        return parameters[idx];
    }

    template <typename T>
    T * getData(size_t idx)
    {
        FParameter * p = getParameter(idx);
        return (T *)p->data;
    }

    void setInput(
        size_t idx,
        void * const data,
        size_t size,
        FParameter::TTL ttl = FParameter::TTL::SINGLE
    )
    {
        setParameter(idx, data, size, FParameter::Type::H2D, ttl);
    }

    void setOutput(
        size_t idx,
        void * const data,
        size_t size,
        FParameter::TTL ttl = FParameter::TTL::USER_MANAGED
    )
    {
        setParameter(idx, data, size, FParameter::Type::D2H, ttl);
    }

    template <typename T>
    void setScalar(
        size_t idx,
        T val
    )
    {
        setParameter(idx, new T(val), sizeof(T), FParameter::Type::VAL, FParameter::TTL::SINGLE);
    }

    template <typename T>
    void setUserInfo(
        size_t idx,
        T * info
    )
    {
        setParameter(idx, info, sizeof(T), FParameter::Type::USR, FParameter::TTL::USER_MANAGED);
    }


    void wait()
    {
        cl::Event::waitForEvents(read_events);
    }

    //*************************************************************************
    //
    // Task Executor public functions
    //
    //*************************************************************************

    void enqueue(FDevice & device, cl::Kernel & kernel, cl::CommandQueue & queue, bool flush = false)
    {
        prepare_buffers(device);
        setKernelArguments(kernel);

        migrate_inputs(queue);
        enqueueTask(kernel, queue);
        migrate_outputs(queue);

        if (flush) queue.flush();
    }

    ~FTask()
    {
        for (auto & p : parameters) {
            if (p->ttl == FParameter::TTL::SINGLE) {
                delete p;
            }
        }
    }
};
