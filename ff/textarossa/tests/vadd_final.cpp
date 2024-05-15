#include <memory>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <ff/ff.hpp>
#include "fnodetask.hpp"
#include "utils.hpp"

using namespace ff;


#define PRINT_RESULTS       false
#define RANDOMIZE_INPUT     false


template <typename T>
T * allocate_array(size_t size)
{
    return aligned_alloc<T>(size);
}

template <typename T>
void generate_array(T * v, size_t size)
{
    const int MAX_VAL = 1024;
    for (size_t i = 0; i < size; ++i) {
    #if RANDOMIZE_INPUT
        v[i] = rand();
    #else
        v[i] = MAX_VAL;
    #endif
    }
}

template <typename T>
void print_array(T * v, size_t size)
{
#if PRINT_RESULTS
    std::cout << "[";
    for (size_t i = 0; i < size; ++i) {
        std::cout << std::setw(4) << v[i] << (i < size - 1 ? ", " : "");
    }
    std::cout << "]\n";
#endif
}

FTask * new_task(int n)
{
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

    int vector_elements;    // size of the vectors
    int number_of_tasks;    // tasks in the stream
    int id;

    generator(int vector_elements, int number_of_tasks, int id = 0)
    : vector_elements(vector_elements)
    , number_of_tasks(number_of_tasks)
    , id(id)
    {}

    void * svc(void * t)
    {
        UNUSED(t);

        for(int i = 0; i < number_of_tasks; i++) {

            int * a = allocate_array<int>(vector_elements);
            int * b = allocate_array<int>(vector_elements);
            int * c = allocate_array<int>(vector_elements);

            generate_array(a, vector_elements);
            generate_array(b, vector_elements);

            int argi = 0;
            FTask * task = new FTask();
            task->setInput(argi++, a, vector_elements * sizeof(int));
            task->setInput(argi++, b, vector_elements * sizeof(int));
            task->setOutput(argi++, c, vector_elements * sizeof(int));
            task->setScalar(argi++, vector_elements);
            task->setUserInfo(argi++, new int(vector_elements));

            ff_send_out((void *) task);
        }
        return EOS;
    }
};

struct drain : public ff_node {

    int id;

    drain(int id = 0)
    : id(id)
    {}

    void * svc(void * t)
    {
        if (t) {
            FTask * task = (FTask *) t;
            task->wait();

            int * c = task->getData<int>(2);
            int * s = task->getData<int>(4);

            print_array(c, *s);

            free(c);

            delete s;
            delete task;
        }
        return(GO_ON);
    }
};

int main(int argc, char * argv[])
{
    std::string bitstream   = "vadd.xclbin";
    std::string kernel_name = "vadd";
    int number_of_workers   = 2;
    int vector_elements     = 1 << 10;
    int number_of_tasks     = 1 << 6;

    if (argc == 1) {
        std::cout << "This program shows the usage of FNodeTask executing a VectorAdd (vadd) kernel containing multiple Compute Units (CUs)." << std::endl;
        std::cout << "\nUsage:\n"
                << "\t" << argv[0] << " file.xclbin [kernel_name] [number_of_workers] [vector_elements] [number_of_tasks]\n"
                << "\nExample:\n"
                << "\t" << argv[0]
                << " " << bitstream
                << " " << kernel_name
                << " " << number_of_workers
                << " $((1 << 10))"
                << " $((1 << 6))"
                << "\n"
                << std::endl;
        return 0;
    }

    int argi = 1;
    if (argc > argi) bitstream         = std::string(argv[argi++]);
    if (argc > argi) kernel_name       = std::string(argv[argi++]);
    if (argc > argi) number_of_workers = atoi(argv[argi++]);
    if (argc > argi) vector_elements   = atoi(argv[argi++]);
    if (argc > argi) number_of_tasks   = atoi(argv[argi++]);

    if (number_of_workers < 0) number_of_workers = 0;
    if (number_of_workers > 8) number_of_workers = 8;

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
    std::vector<std::string> kernel_names;
    for (int i = 0; i < 8; ++i) {
        kernel_names.push_back(kernel_name + ":{" + kernel_name + "_" + std::to_string(i + 1) + "}");
    }

    if (number_of_workers == 0) {
        // The following code implements a pipeline with 3 stages:
        // 1. generator: generates m tasks
        // 2. FNodeTask: executes the "VADD" kernel on the FPGA
        // 3. drain: receives the tasks and frees the memory
        ff_pipeline p;
        p.add_stage(new generator(vector_elements, number_of_tasks));
        p.add_stage(new FNodeTask(device, kernel_names[0]));
        p.add_stage(new drain());

        p.cleanup_nodes();
        p.run_and_wait_end();
        std::cout << "ffTime: " << std::to_string(p.ffTime()) + " ms\n";
    } else {

        // The following code implements a farm with number_of_workers workers,
        // each worker is a pipeline with 2 stages (FNodeTask and drain)
        ff_farm farm;
        farm.add_emitter(new generator(vector_elements, number_of_tasks * number_of_workers));

        std::vector<ff_node *> w;
        for (int i = 0; i < number_of_workers; ++i) {
            ff_pipeline * p = new ff_pipeline();
            p->add_stage(new FNodeTask(device, kernel_names[i]));
            p->add_stage(new drain(i));
            p->cleanup_nodes();
            w.push_back(p);
        }
        farm.add_workers(w);
        farm.remove_collector();
        farm.cleanup_workers();


        auto start = get_time();
        farm.run_and_wait_end();
        auto end = get_time();

        std::cout << "  Time: " + std::to_string(time_diff_ms(start, end)) + " ms\n";
        std::cout << "ffTime: " + std::to_string(farm.ffTime()) + " ms\n";
    }

    return 0;
}
