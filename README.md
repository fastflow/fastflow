[![Build Status](https://travis-ci.com/fastflow/fastflow.svg?branch=master)](https://travis-ci.com/fastflow/fastflow)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![GitHub tag](https://img.shields.io/github/tag/fastflow/fastflow.svg)](http://github.com/fastflow/fastflow/releases)
[![GitHub Issues](https://img.shields.io/github/issues/fastflow/fastflow.svg)](http://github.com/fastflow/fastflow/issues)

# FastFlow: high-performance parallel patterns and building blocks in C++

FastFlow is a programming library implemented in modern C++ and targeting
multi/many-cores (there exists an experimental version based on ZeroMQ targeting
distributed systems). It offers both a set of high-level ready-to-use parallel
patterns and a set of mechanisms and composable components
(called building blocks) to support low-latency and high-throughput data-flow
streaming networks.

FastFlow simplifies the development of parallel applications modelled as a
structured directed graph of processing nodes.
The graph of concurrent nodes is constructed by the assembly of sequential
and parallel building blocks as well as higher-level easy-to-use components
(i.e. parallel patterns) modelling typical schemas of parallel computations
(e.g., pipeline, task-farm, parallel-for, etc.).
FastFlow efficiency stems from the optimized implementation of the base communication
and synchronization mechanisms and from its layered software design.

## Building Blocks

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
*TBC*

## Parallel Patterns
*TBC*

## Building the library
FastFlow is header-only, no need for building.

See the [BUILD.ME](BUILD.ME) file for instructions about building unit tests and examples.

## Supported Platforms
FastFlow is currently actively supported for Linux with gcc >4.8, x86_64 and ARM
Since version 2.0.4, FastFlow is expected to work on any platform with a C++11 compiler. 

### Windows Issues
Windows platform can be particuarly slow in debug mode due to iterator
debugging and secure crt layers. Release mode appers to be not affected by the
issue. See http://blogs.warwick.ac.uk/ahazelden/entry/visual_studio_stl/

The following preprocessor directives /D_SECURE_SCL=0;
/D_HAS_ITERATOR_DEBUGGING=0 (i.e. ADD_DEFINITIONS(-D_SECURE_SCL=0
-D_HAS_ITERATOR_DEBUGGING=0) in cmake) might ameliorate the problem. Consider
however that they cannot be used if the application links libraries compiled
with different options (e.g. Boost).

## FastFlow Maintainer
- Massimo Torquati <torquati@di.unipi.it> (University of Pisa)

## FastFlow History
*TBC*

## How to cite FastFlow
Aldinucci, M. , Danelutto, M. , Kilpatrick, P. and Torquati, M. (2017). Fastflow: High‐Level and Efficient Streaming on Multicore. In Programming multi‐core and many‐core computing systems (eds S. Pllana and F. Xhafa).
[![FF_DOI_badge](https://img.shields.io/badge/DOI-https%3A%2F%2Fdoi.org%2F10.1002%2F9781119332015.ch13-blue.svg)](https://doi.org/10.1002/9781119332015.ch13)
