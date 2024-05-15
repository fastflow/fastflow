#pragma once

#include "ff/ff.hpp"
#include "fdevice.hpp"
#include "ftask.hpp"

#include <string>

class FNodeTask : public ff::ff_node
{
private:
    FDevice & device;
    std::string kernel_name;
    cl::CommandQueue & queue;
    cl::Kernel kernel;

public:
    FNodeTask(FDevice & device, std::string kernel_name)
    : device(device)
    , kernel_name(kernel_name)
    , queue(device.queue_)
    , kernel(device.createKernel(kernel_name))
    {}

    void * svc (void * t)
    {
        if (t) {
            FTask * task = (FTask *)t;
            task->enqueue(device, kernel, queue, false);
        }
        return t;
    }
};
