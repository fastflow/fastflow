/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */


/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

#ifndef FF_CONFIG_HPP
#define FF_CONFIG_HPP

#include <cstddef> 
#include <climits>
#if defined(TRACE_FASTFLOW)
#include <iostream>
#endif


#if !defined(CACHE_LINE_SIZE)
#define CACHE_LINE_SIZE 64
#endif

/*
 * NOTE: by default FF_BOUNDED_BUFFER is not defined
 * because the uSWSR_Ptr_Buffer may act as a bounded queue.
 */
#if defined(FF_BOUNDED_BUFFER)
#define FFBUFFER SWSR_Ptr_Buffer
#else  // unbounded buffer
#define FFBUFFER uSWSR_Ptr_Buffer
#endif

/* To save energy and improve hyperthreading performance
 * define the following macro
 */
//#define SPIN_USE_PAUSE 1

/* To enable OPENCL support
 *
 */
//#define FF_OPENCL 1 


/* To enable task callbacks
 *
 * If enabled, 2 callbacks are called by the run-time:
 *  - one before receiving the task in input
 *  - one just after having computed the task (before sending it out)
 */
//#define FF_TASK_CALLBACK 1

namespace ff {
static const size_t FF_EOS           = (ULLONG_MAX);  /// automatically propagated
static const size_t FF_EOS_NOFREEZE  = (FF_EOS-0x1);  /// non automatically propagated
static const size_t FF_EOSW          = (FF_EOS-0x2);  /// propagated only by farm's stages
static const size_t FF_GO_ON         = (FF_EOS-0x3);  /// non automatically propagated
static const size_t FF_GO_OUT        = (FF_EOS-0x4);  /// non automatically propagated
static const size_t FF_BLK           = (FF_EOS-0x5);  /// automatically propagated
static const size_t FF_NBLK          = (FF_EOS-0x6);  /// automatically propagated
static const size_t FF_TAG_MIN       = (FF_EOS-0xa);  /// just a lower bound mark
// The FF_GO_OUT is quite similar to the FF_EOS_NOFREEZE, both are not propagated automatically but while 
// the first one is used to exit the main computation loop and in case being freezed, the second one is used 
// to exit the computation loop and keep spinning on the input queue for a new task without being freezed.
// EOSW is like EOS but it is not propagated outside a farm pattern. If an emitter receives EOSW in input,
// than it will be discarded.
// FF_BLK enables blocking mode, FF_NBLK disable blocking mode (aka nonblocking). 
//
}

#if defined(TRACE_FASTFLOW)
#define FFTRACE(x) x
#else
#define FFTRACE(x)
#endif

#if defined(BLOCKING_MODE)
#define RUNTIME_MODE true
#else
#define RUNTIME_MODE false   // by default the run-time is in nonblocking mode
#endif

// the barrier implementation to use
#if !defined(BARRIER_T)
#define BARRIER_T             spinBarrier
#endif

// maximum number of threads that can be spawned
#if !defined(MAX_NUM_THREADS)
#define MAX_NUM_THREADS       512 
#endif

// maximum number of workers in a farm
#define DEF_MAX_NUM_WORKERS   (MAX_NUM_THREADS-2)

// NOTE: BACKOFF_MIN/MAX are lower and upper bound backoff values.
// Notice that backoff bounds are highly dependent from the system and 
// from the concurrency levels. This values should be carefully tuned
// in order to achieve the maximum performance.

#if !defined(BACKOFF_MIN)
#define BACKOFF_MIN 128
#endif
#if !defined(BACKOFF_MAX)
#define BACKOFF_MAX 1024
#endif

// TODO:
//#if defined(NO_CMAKE_CONFIG)

// TODO change to __GNUC__ that is portable. GNUC specific code currently works
// on linux only
#if defined(__USE_GNU) //linux
//#if defined(__GNUC__) 
#define HAVE_PTHREAD_SETAFFINITY_NP 1
//#warning "Is GNU compiler"
#endif 

#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#define MAC_OS_X_HAS_AFFINITY (defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && (__MAC_OS_X_VERSION_MIN_REQUIRED >= 1050))
//defined (MAC_OS_X_VERSION_10_5) && MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_5)
#endif

//#else
// the config.h file will be generated by cmake
//#include <ff/config.h>
//#endif // NO_CMAKE_CONFIG

#if defined(USE_CMAKE_CONFIG) && !defined(NOT_USE_CMAKE_CONFIG)
#include <cmake.modules/ffconfig.h>
#endif

// OpenCL additional code needed to compile kernels
#define FF_OPENCL_DATATYPES_FILE "ff_opencl_datatypes.cl"

#endif /* FF_CONFIG_HPP */
