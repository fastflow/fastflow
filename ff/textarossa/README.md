# FFpga
Experimental [`FastFlow`](https://github.com/fastflow/fastflow) node called `FNodeTask` to offload computation of `Vitis HLS` kernels on Alveo FPGAs.
The `FNodeTask` can be used in any place where you need an `ff_node`/`ff_node_t`.
It offloads tasks to pre-compiled kernels on an Alveo FPGA

## Sample code
The `vadd_cus` program executes a VectorAdd (`vadd`) kernel composed of multiple Compute Units (CUs).
This program will execute a `ff_pipeline` composed of a `generator` node that creates tasks to feed the `FNodeTask` node, and finally a `drain` node that receives computed results.
The `vadd_pipe` is similar to `vadd_cus`, but put two instances of `vadd` kernel in pipeline.

Both programs can run a `ff_farm` that wraps the pipeline described above, and can be run by specifying `num_workers` greater than 0 (max 8 pipeline instances for `vadd_cus`, and 4 for `vadd_pipe).

## Compile `vadd` kernel
```bash
cd kernels/vadd_cus
make all TARGET=hw # sw_emu or hw_emu
```

## Compile host
```bash
cd test
make vadd_cus		# compiles the vadd_cus host program
make vadd_pipe		# compiles the vadd_pipe host program
```

## Run
The `Makefile` contains the following tests:

```bash
$ make test_pipe	# runs a pipeline containing FNodeTask
$ make test_farm	# runs a farm of pipeline containing FNodeTask
$ make test_pipe2	# runs a pipeline containing 2 FNodeTask
$ make test_farm2	# runs a farm of pipeline containing 2 FNodeTask
```

You can also compile and run both host programs as follows:

```
Usage:
        ./vadd_cus file.xclbin [kernel_name] [num_workers] [chain_tasks] [vec_elems] [vec_nums]
Example:
        ./vadd_cus vadd.xclbin vadd 2 0 $((1 << 10)) $((1 << 6))
```