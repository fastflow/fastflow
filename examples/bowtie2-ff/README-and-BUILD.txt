July 9, 2014

Bowtie2-ff is bowtie2.0.6 porting on the FastFlow programming
framework. http://mc-fastflow.sourceforge.net/

FastFlow is licenced under LGPLv3. See FastFlow licence file.

Bowtie2-ff is part of the FastFlow examples, and it is licenced under
GNU GPL as the original Bowtie-2.0.6 package.

The implementation proposed is based on the Fastflow Master-Worker
pattern (lock-free and fence-free). The implementation  takes
advantage of the FastFlow pattern flexibility to define a memory-affine
self-balancing scheduling policy for reads processing. Bowtie2-ff
offers two levels of memory allocation and 
accesses optimisations. Each level is provided with workload
dynamically partitioned among workers. These optimisations consists
of:  

1) Master-Worker with threads pinning and memory affinity for
threads' private data;
2) Master-Worker with interleaved allocation policy among memory nodes for shared data (genome); These two features can be defined at compile time.

Implementation is documented and experimented for performance against
Bowtie-2.0.6 and Bowtie-2.2.1 in:

1) C. Misale, G. Ferrero, M. Torquati, and M. Aldinucci, “Sequence
   alignment tools: one parallel pattern to rule them all?,” BioMed
   Research International, 2014. doi:10.1155/2014/539410

2) C. Misale, “Accelerating Bowtie2 with a lock-less concurrency approach
   and memory affinity,” in Proc. of Intl. Euromicro PDP 2014:
   Parallel Distributed and network-based Processing, Torino, Italy,
   2014. doi:10.1109/PDP.2014.50  
  

1. HOW TO BUILD

Compile with `make'.  To specify the FastFlow directory, modify the
`FF_DIR' variable located in the Makefile.  To use threads pinning and
memory affinity, uncomment the `-DFF_NUMA_ALLOC' flag present in the
Makefile.  To use interleaved genome allocation, uncomment the
`-DFF_INDEX_INTERLEAVED' flag present in the Makefile.

Compile with `cmake'.

On the fastflow root directory:

>mkdir build
>cd build
>cmake ..
(look for possible errors and warnings)
>cd examples/bowtie2-ff
>make
(both bowtie2 and bowtie2-ff will be built)


2. HOW TO SPECIFY CPUs LIST FOR PINNING

At the moment, the list of CPUs for pinning must be defined directly into 
the source code.  The automatic definition of CPUs lists is a work in progress.
In `bt2_search.cpp' are present two examples of string lists of CPUs.
A CPUs list is an array of chars of CPU IDs.
Threads will be mapped on CPUs following the specified order.
To get informations about CPUs and NUMA nodes, use the `lscpu' command.

3. EXECUTE BOWTIE
Bowtie2 can be executed with parameters specified by the manual. 
No additional parameter is required. 
To execute Bowtie2 on the example sets, type:

>./bowtie2 -t -p <threads_num> -x example/index/lambda_virus -U example/reads/longreads.fq -S output.sam
		
for a single-end alignment with default options.

Type:

>./bowtie2 -t -p <threads_num> -x example/index/lambda_virus -1 example/reads/reads_1.fq -2 example/reads/reads_2.fq -S output.sam
		
for a paired-end alignment with default options.

4. SUPPORTED ARCHITECTURES

Building Bowtie 2 from source requires POSIX environment with a c++ compiler and make.
Support for -std=c++11 or -std=c++0x is required for the FastFlow version (the code is likely to work also with an older compiler, but has not been tested). Libnuma is required to access advanced memory affinity features.

Bowtie2-ff has been extensively tested on:

- Linux (gcc 4.7/4.8/4.9)
- Mac OS 10.9.x with gcc 4.8 and Clang 5.1

FastFlow is also supported on Windows 7/8 with Visual Studio. The code
is likely to work also on Windows 7/8 with Visual Studio 2010/2012
Win64, but has not been tested (minor code fixes might be required). 

Main FastFlow development platform are Linux and MacOS.


Claudia Misale (misale@di.unito.it)
Marco Aldinucci (aldinuc@di.unito.it)

Parallel programming models group at University of Torino
http://alpha.di.unito.it

