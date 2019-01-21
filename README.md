[![Build Status](https://travis-ci.com/fastflow/fastflow.svg?branch=master)](https://travis-ci.com/fastflow/fastflow)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![GitHub tag](https://img.shields.io/github/tag/fastflow/fastflow.svg)](http://github.com/fastflow/fastflow/releases)
[![GitHub Issues](https://img.shields.io/github/issues/fastflow/fastflow.svg)](http://github.com/fastflow/fastflow/issues)

# FastFlow: high-performance parallel patterns in C++

FastFlow is a multi-core programming framework implemented as a C++ template 
library that offers a set of mechanisms to support low-latency and 
high-bandwidth data flows in a network of threads running on
a cache-coherent multi-core architectures. On these architectures, one of the 
key performance issues concern memory fences, which are required to keep the 
various caches coherent. 
FastFlow provides the programmer with two basic mechanisms:
  1. efficient point-to-pint communication channels;
  2. a memory allocator.

Communication channels, as typical is in streaming applications, are 
unidirectional and asynchronous. They are implemented via fence-free FIFO
queues. The memory allocator is built on top of these queues, thus taking 
advantage of their efficiency.
On top of these basic machnisms FastFlow provides a library of explicitly 
parallel constructs (a.k.a. skeletons) such as pipeline and farm.
The farm skeleton, exploits functional replication and abstracts the 
parallel filtering of successive independent items of the stream under the 
control of a scheduler.


## Building

FastFlow is header-only, no need for building.

See the [BUILD.ME](BUILD.ME) file for instructions about building unit tests and examples.

## Supported Platforms

FastFlow is currently actively supported for:

- Linux with gcc >4.8 x86_64
- Windows >=7 with MSVS >=2013
- Mac OS >=10.9 with gcc >4.8 or clang >=5

Although not officially supported (yet), FastFlow has been tested on:
- Linux/PPC with gcc
- Linux/ARM with gcc

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

## FastFlow Team
- Massimo Torquati <torquati@di.unipi.it> (maintainer)
- Marco Aldinucci  <aldinuc@di.unito.it> (co-maintainer)
- Maurizio Drocco <maurizio.drocco@gmail.com> (co-maintainer)

## How to cite FastFlow
Aldinucci, M. , Danelutto, M. , Kilpatrick, P. and Torquati, M. (2017). Fastflow: High‐Level and Efficient Streaming on Multicore. In Programming multi‐core and many‐core computing systems (eds S. Pllana and F. Xhafa).
[![FF_DOI_badge](https://img.shields.io/badge/DOI-https%3A%2F%2Fdoi.org%2F10.1002%2F9781119332015.ch13-blue.svg)](https://doi.org/10.1002/9781119332015.ch13)
