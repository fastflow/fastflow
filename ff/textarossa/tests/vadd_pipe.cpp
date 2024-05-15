// #include <memory>
// #include <iostream>
// #include <sstream>

// #include <ff/ff.hpp>
// #include "fnodetask.hpp"

// using namespace ff;

// int bank_in1(int kernel_id) { return kernel_id * 4; }
// int bank_in2(int kernel_id) { return bank_in1(kernel_id) + 2; }
// int bank_out(int kernel_id) { return bank_in1(kernel_id); }

// // This function helps the generation of a new task.
// // It allocates the memory for the task, fills the input arrays with some data
// // and returns the task.
// // The FTask object is a wrapper around the task that contains the input and
// // output arrays, the scalar values and the information about the memory banks
// // used by the task.
// // The order of parameters is crucial and must be in the same order as the
// // signature of the kernel function.
// FTask * new_task(int n, int kernel_id)
// {
//     const int MAX_VAL = 1024;

//     size_t size_in_bytes = n * sizeof(int);
//     int ret = 0;

//     int * a = nullptr;
//     int * b = nullptr;
//     int * c = nullptr;
//     int * s = nullptr;

//     ret |= posix_memalign((void **)&a, 4096, size_in_bytes);
//     ret |= posix_memalign((void **)&b, 4096, size_in_bytes);
//     ret |= posix_memalign((void **)&c, 4096, size_in_bytes);
//     s = (int *) malloc(sizeof(int));

//     if (ret != 0) {
//         std::cerr << "ERROR: failed to allocate aligned memory!\n";
//         exit(-1);
//     }

//     for (int i = 0; i < n; ++i) a[i] = i; // rand() % MAX_VAL;
//     for (int i = 0; i < n; ++i) b[i] = i; // rand() % MAX_VAL;
//     *s = n;

//     FTask * task = new FTask();
//     task->add_input(a,  size_in_bytes, bank_in1(kernel_id));
//     task->add_input(b,  size_in_bytes, bank_in2(kernel_id));
//     task->add_output(c, size_in_bytes, bank_out(kernel_id));
//     task->add_scalar(s, sizeof(int));

//     return task;
// }

// // This function helps the generation of a new task for the "VINC" kernel.
// FTask * new_task_middle(int n, int kernel_id, int * cc)
// {
//     const int MAX_VAL = 1024;

//     size_t size_in_bytes = n * sizeof(int);
//     int ret = 0;

//     int * a = cc;
//     int * b = cc;
//     int * c = nullptr;
//     int * s = nullptr;

//     ret |= posix_memalign((void **)&c, 4096, size_in_bytes);
//     s = (int *) malloc(sizeof(int));

//     if (ret != 0) {
//         std::cerr << "ERROR: failed to allocate aligned memory!\n";
//         exit(-1);
//     }
//     *s = n;

//     FTask * task = new FTask();
//     task->add_input(a,  size_in_bytes, bank_in1(kernel_id));
//     task->add_input(b,  size_in_bytes, bank_in2(kernel_id));
//     task->add_output(c, size_in_bytes, bank_out(kernel_id));
//     task->add_scalar(s, sizeof(int));

//     return task;
// }

// // This ff_node is used in the farm implementation to start the execution of
// // the workes (pipeline containing the FPGA kernels).
// struct fake_emitter : public ff_node {
//     int n;  // number of generators

//     fake_emitter(int n) : n(n) {}

//     void * svc(void * t)
//     {
//         for (int i = 0; i < n; ++i) {
//             ff_send_out((int *)(0xBEEF), i);
//         }
//         return EOS;
//     }
// };

// // This ff_node generates the tasks.
// struct generator : public ff_node {

//     int n;    // size of the vectors
//     int m;    // tasks in the stream
//     int index;

//     generator(int n, int m, int index = 0)
//     : n(n)
//     , m(m)
//     , index(index)
//     {}

//     void * svc(void * t)
//     {
//         for(int i = 0; i < m; i++) {
//             // std::cout << "generator " + std::to_string(index) + ": generating task " + std::to_string(i) + "\n";
//             FTask * task = new_task(n, index);
//             ff_send_out((void *) task);
//         }
//         return EOS;
//     }
// };

// // This ff_node receivese the result of "VADD" tasks and generate the "VINC" tasks.
// class middle : public ff_node {
// public:

//     int n;
//     int index;

//     int i;
//     std::vector<FTask *> tasks;

//     middle(int n, int index)
//     : n(n)
//     , index(index)
//     , i(0)
//     {}

//     void * svc(void * t)
//     {
//         if (t) {
//             // std::cout << "middle " + std::to_string(index) + ": receiving task " + std::to_string(i) + "\n";
//             FTask * task = (FTask *) t;
//             int * c = (int *)task->outputs[0].ptr;

//             free(task->inputs[0].ptr);
//             free(task->inputs[1].ptr);
//             task->outputs[0].ptr = nullptr;
//             free(task->scalars[0].ptr);
//             tasks.push_back(task);

//             FTask * newTask = new_task_middle(n, index, c);
//             ff_send_out(newTask);

//             i++;
//         }
//         return(GO_ON);
//     }

//     void svc_end()
//     {
//         for (auto * t : tasks) {
//             delete t;
//         }
//     }
// };

// // This ff_node receives the result of "VINC" tasks.
// struct drain : public ff_node {

//     int index;
//     int i;

//     std::vector<FTask *> tasks;

//     drain(int index = 0)
//     : index(index)
//     , i(0)
//     {}

//     void * svc(void * t)
//     {
//         if (t) {
//             FTask * task = (FTask *) t;
//             // std::cout << "drain " + std::to_string(index) + ": receiving task " + std::to_string(i++) + "\n";
//             free(task->inputs[0].ptr);
//             // free(task->inputs[1].ptr);
//             free(task->outputs[0].ptr);
//             free(task->scalars[0].ptr);
//             tasks.push_back(task);
//         }
//         return(GO_ON);
//     }

//     void svc_end()
//     {
//         for (auto * t : tasks) {
//             delete t;
//         }
//     }
// };

// int main(int argc, char * argv[])
// {
//     std::string bitstream   = "vadd.xclbin";
//     std::string kernel_name = "vadd";
//     int nWorkers   = 2;
//     bool chain     = false;
//     int n          = 1 << 10;
//     int m          = 1 << 6;

//     if (argc == 1) {
//         std::cout << "This program shows the usage of FNodeTask executing a VectorAdd (vadd) kernel containing multiple Compute Units (CUs)." << std::endl;
//         std::cout << "\nUsage:\n"
//                 << "\t" << argv[0] << " file.xclbin [kernel_name] [num_workers] [chain_tasks] [vec_elems] [vec_nums]\n"
//                 << "\nExample:\n"
//                 << "\t" << argv[0]
//                 << " " << bitstream
//                 << " " << kernel_name
//                 << " " << nWorkers
//                 << " " << (chain ? "1" : "0")
//                 << " $((1 << 10))"
//                 << " $((1 << 6))"
//                 << "\n"
//                 << std::endl;
//         return 0;
//     }

//     int argi = 1;
//     if (argc > argi) bitstream   = std::string(argv[argi++]);
//     if (argc > argi) kernel_name = std::string(argv[argi++]);
//     if (argc > argi) nWorkers    = atoi(argv[argi++]);
//     if (argc > argi) chain       = atoi(argv[argi++]) > 0;
//     if (argc > argi) n           = atoi(argv[argi++]);
//     if (argc > argi) m           = atoi(argv[argi++]);

//     if (nWorkers < 0) nWorkers = 0;
//     if (nWorkers > 4) nWorkers = 4;

//     size_t size_in_bytes = n * sizeof(int);
//     size_t size_in_kb = size_in_bytes / 1024;
//     std::cout << "Executing " << kernel_name << " with " << bitstream + "\n";
//     std::cout << "Generating " << (m * nWorkers) << " tasks in total and using " << nWorkers << " workers!\n";
//     std::cout << "Task (" << size_in_kb << " KB, " << size_in_kb << " KB) -> (" << size_in_kb << " KB)" << "\n\n";
//     std::cout << "nWokers = " << nWorkers << "\n"
//               << "  chain = " << chain    << "\n"
//               << "      n = " << n        << "\n"
//               << "      m = " << m        << "\n"
//               << std::endl;

//     FDevice device(bitstream);
//     std::vector<std::string> kernel_names;
//     for (int i = 0; i < 8; ++i) {
//         kernel_names.push_back(kernel_name + ":{" + kernel_name + "_" + std::to_string(i + 1) + "}");
//     }

//     if (nWorkers == 0) {

//         // The following code implements a pipeline with 5 stages:
//         // 1. generator: generates n tasks
//         // 2. FNodeTask: executes the kernel "VADD" on the FPGA
//         // 3. middle:    receives the output of "VADD" and generates a new task
//         //               for the "VINC" kernel
//         // 4. FNodeTask: executes the kernel "VINC" on the FPGA
//         // 5. drain:     receives the output of "VINC" and frees the memory
//         ff_pipeline p;
//         p.add_stage(new generator(n, m));
//         p.add_stage(new FNodeTask(device, kernel_names[0], chain));
//         p.add_stage(new middle(n, 1));
//         p.add_stage(new FNodeTask(device, kernel_names[1], chain));
//         p.add_stage(new drain());

//         p.cleanup_nodes();
//         p.run_and_wait_end();
//         std::cout << "ffTime: " << std::to_string(p.ffTime()) + " ms\n";
//     } else {
//         // The following code implements a farm with "nWorkers" workers,
//         // each worker has a pipeline with 5 stages like the one above.
//         ff_farm farm;
//         farm.add_emitter(new fake_emitter(nWorkers));

//         std::vector<ff_node *> w;
//         for (int i = 0; i < nWorkers; ++i) {
//             ff_pipeline * p = new ff_pipeline();
//             p->add_stage(new generator(n, m, i));
//             p->add_stage(new FNodeTask(device, kernel_names[i], chain));
//             p->add_stage(new middle(n, i * 2 + 1));
//             p->add_stage(new FNodeTask(device, kernel_names[i * 2 + 1], chain));
//             p->add_stage(new drain());
//             p->cleanup_nodes();
//             w.push_back(p);
//         }
//         farm.add_workers(w);
//         farm.remove_collector();
//         farm.cleanup_workers();


//         auto start = get_time();
//         farm.run_and_wait_end();
//         auto end = get_time();

//         std::cout << "  Time: " + std::to_string(time_diff_ms(start, end)) + " ms\n";
//         std::cout << "ffTime: " + std::to_string(farm.ffTime()) + " ms\n";
//         // farm.ffStats(std::cout);
//     }

//     return 0;
// }
