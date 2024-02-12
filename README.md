[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub tag](https://img.shields.io/github/tag/fastflow/fastflow.svg)](http://github.com/fastflow/fastflow/releases)
[![GitHub Issues](https://img.shields.io/github/issues/fastflow/fastflow.svg)](http://github.com/fastflow/fastflow/issues)

# FastFlow: high-performance parallel patterns and building blocks in C++

FastFlow is a programming library implemented in modern C++ targeting
multi/many-cores and distributed systems (the distributed run-time is experimental).
It offers both a set of high-level ready-to-use parallel patterns and a set
of mechanisms and composable components (called building blocks) to support low-latency and high-throughput data-flow streaming networks.

FastFlow simplifies the development of parallel applications modelled as a
structured directed graph of processing nodes.
The graph of concurrent nodes is constructed by the assembly of sequential
and parallel building blocks as well as higher-level parallel patterns modelling typical schemas of parallel computations (e.g., pipeline, task-farm, parallel-for, etc.).
FastFlow efficiency stems from the optimized implementation of the base communication and synchronization mechanisms and from its layered software design.

## FastFlow's Building Blocks

FastFlow nodes represent sequential computations executed by a dedicated thread.
A node can have zero, one or more input channels and zero, one or more output channels.
As typical is in streaming applications, communication channels are unidirectional and
asynchronous. They are implemented through Single-Producer Single-Consumer
(SPSC) FIFO queues carrying memory pointers. Operations on such queues (that can have either
bounded or unbounded capacity) are based on  non-blocking lock-free synchronization protocol.
To promote power-efficiency vs responsiveness of the nodes, a blocking concurrency
control operation mode is also available.

The semantics of sending data references over a communication channel is that of transferring
the ownership of the data pointed by the reference from the sender node (producer) to the
receiver node (consumer) according to the producer-consumer model.
The data reference is de facto a capability, i.e. a logical token that grants access to a given
data or to a portion of a larger data structure. Based on this reference-passing semantics,
the receiver is expected to have exclusive access to the data reference received from one of
the input channels, while the producer is expected not to use the reference anymore.

The set of FastFlow building blocks is:

**node**. This is the basic abstraction of the building blocks. It defines the unit of sequential execution in the FastFlow library. A node encapsulates either user’s code (i.e. business logic) or RTS code. User’s code can also be wrapped by a FastFlow node executing RTS code to manipulate and filter input and output data before and after the execution of the business logic code. Based on the number of input/output channels it is possible to distinguish three different kinds of sequential nodes: *standard node* with one input and one output channel, *multi-input* with many inputs and one output channel, and finally *multi-output* with one input and many outputs. 
A generic node performs a loop that: i) gets a data item (through a memory reference to a data structure) from one of its input queues; ii) executes a functional code working on the data item and possibly on a state maintained by the node itself by calling its service method svc(); iii) puts a memory reference to the resulting item(s) into one or multiple output queues selected according to a predefined or user-defined policy.

**node combiner**. It allows the user to combine two nodes into one single sequential node. Conceptually, the operation of combining sequential nodes is similar to the composition of two functions. In this case, the functions are the service functions of the two nodes (e.g., the *svc* method). This building block promotes code reuse through fusion of already implemented nodes and it can also be used to reduce the threads used to run the data-flow network by executing the functions of multiple nodes by a single thread.

**pipeline**. The pipeline allows building blocks to be connected in a linear chain. It is used both as a container of building blocks as well as an application topology builder. At execution time, the pipeline building block models the data-flow execution of its building blocks on data elements flowing in a streamed fashion.

**farm**. It models functional replication of building blocks coordinated by a master node called Emitter. The simplest form is composed of two computing entities executed in parallel: a multi-output master node (the *Emitter*), and a pool of pipeline building blocks called *Workers*. The Emitter node schedules the data elements received in input to the Workers using either a default policy (i.e. *round-robin* or *on-demand*) or according to the algorithm implemented by the user code defined in its service method. In this second scenario, the stream elements scheduling is controlled by the user through a custom policy.

**All-to-All** The All-to-All (briefly **A2A**) building block defines two distinct sets of Workers connected accordig to the *shuffle communication pattern*. This means that each Worker in the first set (called *L-Worker*) is connected to all the Workers in the second set (called *R-Workers*). The user may implement any custom distribution policy in the L-Workers (e.g., sending each data item to a specific worker of the R-Worker set, broadcasting data elements, executing a *by-key* routing, etc). The default distribution policy is *round-robin*.

A brief description of the FastFlow building block software layer can be found [here](https://docs.google.com/presentation/d/1mCJ9Bf4zo3MX2DFGG0zfbJ2URdCIJoECt87-Rkt2swc/edit?usp=sharing).

## Available Parallel Patterns

In FastFlow, all parallel patterns available are implemented on top of building blocks. 
Parallel Patterns are parametric implementations of well-known structures suitable 
for parallelism exploitation. The high-level patterns currently available in FastFlow library are: 
**ff_Pipe**, **ff_Farm/ff_OFarm**, **ParallelFor/ParallelForReduce/ParallelForPipeReduce**, **poolEvolution**,
**ff_Map**, **ff_mdf**, **ff_DC**, **ff_stencilReduce**. 

Differenting from the building block layer, the parallel patterns layer is in continuous evolution. 
As soon as new patterns are recognized or new smart implementations are available for the existing patterns, 
they are added to the high-level layer and provided to the user.


## Building the library

FastFlow is a header-only library, for the shared-memory run-time, there are basically no dependencies
(but remember to run the script mapping_string.sh in the ff directory!).
For the distributed-memory run-time, you need to install:
 - Cereal for (automatic) serialization/deserialization purposes (https://uscilab.github.io/cereal/)
 - OpenMPI for experimenting with the MPI communication back-end (https://www.open-mpi.org/software/ompi)

While Cereal is mandatory, OpenMPI installation is optional and can be disabled at compile-time by compiling the
code with '-DDFF_EXCLUDE_MPI' (or make EXCLUDE_MPI=1). To compile the tests with the distributed run-time you need a
recent compiler supporting the -std=c++20 standard (e.g., gcc 10 or above).
In addition, by default the *shared-memory* version uses the non-blocking concurrency control mode, wherease the
*distributed version* uses the blocking mode for its run-time system. You can control the concurrency control mode
either at compile time (see the config.hpp file) or at run-time by calling the proper methods before running the application.

See the [BUILD.ME](BUILD.ME) file for instructions about building unit tests and examples.
NOTES: currently, the cmake-based compilation of distributed tests has been disabled. 

## Supported Platforms
FastFlow is currently actively supported for Linux with gcc >4.8, x86_64 and ARM
Since version 2.0.4, FastFlow is expected to work on any platform with a C++11 compiler. 

## FastFlow Maintainer
Massimo Torquati (University of Pisa)
<torquati@di.unipi.it> <massimo.torquati@unipi.it>

## FastFlow History
The FastFlow project started in the beginning of 2010 by Massimo Torquati (University of Pisa) and 
Marco Aldinucci (University of Turin). 
Over the years several other people (mainly from the Parallel Computing Groups of the University of Pisa and Turin) contributed with ideas and code to the development of the project. FastFlow has been used
as run-time system in three EU founded research projects: ParaPhrase, REPARA and RePhrase. Currently is one of the tools used in the Euro-HPC project TEXTAROSSA.

More info about FastFlow and its parallel building blocks can be found here:
Massimo Torquati (Pisa, PhD Thesis) "Harnessing Parallelism in Multi/Many-Cores with Streams and Parallel Patterns"

## About the License
From version 3.0.1, FastFlow is released with a dual license: <strong>LGPL-3</strong> and <strong>MIT</strong>. 

## How to cite FastFlow
Aldinucci, M. , Danelutto, M. , Kilpatrick, P. and Torquati, M. (2017). Fastflow: High‐Level and Efficient Streaming on Multicore. In Programming multi‐core and many‐core computing systems (eds S. Pllana and F. Xhafa).
[![FF_DOI_badge](https://img.shields.io/badge/DOI-https%3A%2F%2Fdoi.org%2F10.1002%2F9781119332015.ch13-blue.svg)](https://doi.org/10.1002/9781119332015.ch13)
