#include <memory>
#include <iostream>
#include <sstream>

#include <ff/ff.hpp>
#include "fnodetask.hpp"

using namespace ff;

FTask * new_task(int n)
{
    const int MAX_VAL = 1024;

    size_t size_in_bytes = n * sizeof(int);
    int ret = 0;

    int * a = nullptr;
    int * b = nullptr;
    int * c = nullptr;
    int * s = nullptr;

    ret |= posix_memalign((void **)&a, 4096, size_in_bytes);
    ret |= posix_memalign((void **)&b, 4096, size_in_bytes);
    ret |= posix_memalign((void **)&c, 4096, size_in_bytes);
    s = (int *) malloc(sizeof(int));

    if (ret != 0) {
        std::cerr << "ERROR: failed to allocate aligned memory!\n";
        exit(-1);
    }

    for (int i = 0; i < n; ++i) a[i] = i; // rand() % MAX_VAL;
    for (int i = 0; i < n; ++i) b[i] = i; // rand() % MAX_VAL;
    *s = n;

    FTask * task = new FTask();
    task->setParameter(0, a, size_in_bytes, FParameter::Type::H2D);
    task->setParameter(1, b, size_in_bytes, FParameter::Type::H2D);
    task->setParameter(2, c, size_in_bytes, FParameter::Type::D2H);
    task->setParameter(3, s, sizeof(int), FParameter::Type::VAL);

    return task;
}

// This ff_node generates the tasks.
struct generator : public ff_node {

#define GENERATE_UPFRONT 0

    int vector_elements;    // size of the vectors
    int number_of_tasks;    // tasks in the stream
    int id;
    std::vector<FTask *> tasks;

    generator(int vector_elements, int number_of_tasks, int id = 0)
    : vector_elements(vector_elements)
    , number_of_tasks(number_of_tasks)
    , id(id)
    {
        #if GENERATE_UPFRONT
        for(int i = 0; i < number_of_tasks; i++) {
            std::cout << "generator " + std::to_string(id) + ": generating task " + std::to_string(i) + "\n";
            tasks.push_back(new_task(vector_elements));
        }
        #endif
    }

    void * svc(void * t)
    {
        (void)t; // unused

        for(int i = 0; i < number_of_tasks; i++) {
            #if GENERATE_UPFRONT
            ff_send_out((void *) tasks[i]);
            #else
            std::cout << "generator " + std::to_string(id) + ": generating task " + std::to_string(i) + "\n";
            FTask * task = new_task(vector_elements);
            ff_send_out((void *) task);
            #endif
        }
        return EOS;
    }
};

// This ff_node receives the tasks and frees the memory at the end.
struct drain : public ff_node {

    int id;
    int i;

    drain(int id = 0)
    : id(id)
    , i(0)
    {}

    void * svc(void * t)
    {
        if (t) {
            FTask * task = (FTask *) t;
            task->wait();

            // print "c" vector
            auto * c = (int *) task->getParameter(2)->data;
            auto * s = (int *) task->getParameter(3)->data;
            // std::cout << "c = [";
            // for (int i = 0; i < *s; ++i) {
            //     std::cout << c[i] << ", ";
            // }
            // std::cout << "]\n";

            std::cout << "drain " + std::to_string(id) + ": receiving task " + std::to_string(i++) + "\n";
            auto * a = (int *) task->getParameter(0)->data;
            auto * b = (int *) task->getParameter(1)->data;
            // auto * c = (int *) task->getParameter(2)->data;
            // auto * s = (int *) task->getParameter(3)->data;
            // free(a);
            // free(b);
            // free(c);
            // free(s);

            delete task;
        }
        return(GO_ON);
    }
};

int main(int argc, char * argv[])
{
    std::string bitstream   = "krnl_vadd.xclbin";
    std::string kernel_name = "krnl_vadd";
    int number_of_workers   = 1;
    int vector_elements     = 8;
    int number_of_tasks     = 8;

    size_t size_in_bytes = vector_elements * sizeof(int);
    size_t size_in_kb = size_in_bytes / 1024;
    std::cout << "Executing " << kernel_name << " with " << bitstream + "\n";
    std::cout << "Generating " << (number_of_tasks * number_of_workers) << " tasks in total and using " << number_of_workers << " workers!\n";
    std::cout << "Task (" << size_in_kb << " KB, " << size_in_kb << " KB) -> (" << size_in_kb << " KB)" << "\n\n";
    std::cout << "nWokers         = " << number_of_workers << "\n"
              << "vector_elements = " << vector_elements   << "\n"
              << "number_of_tasks = " << number_of_tasks   << "\n"
              << std::endl;

    FDevice device(bitstream);

        ff_pipeline p;
        p.add_stage(new generator(vector_elements, number_of_tasks));
        p.add_stage(new FNodeTask(device, kernel_name));
        p.add_stage(new drain());

        p.cleanup_nodes();
        p.run_and_wait_end();
        std::cout << "ffTime: " << std::to_string(p.ffTime()) + " ms\n";

    return 0;
}
