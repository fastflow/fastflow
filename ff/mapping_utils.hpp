/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef __MAPPING_UTILS_HPP_
#define __MAPPING_UTILS_HPP_
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License version 3 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *
 ****************************************************************************
 */

/* Support for thread pinning and mapping
 * Linux and MacOS are supported
 * 
 * Author: Massimo Aldinucci, Massimo Torquati
 * Revision: beta
 * Date: Oct 5, 2010
 *
 */

#include <iostream>
#if defined(__linux__)
#include <sys/types.h>
#include <sys/resource.h>
#include <asm/unistd.h>
#define gettid() syscall(__NR_gettid)
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h> 
#include <mach/mach_init.h>
#include <mach/thread_policy.h> 
// If you don't include mach/mach.h, it doesn't work.
// In theory mach/thread_policy.h should be enough
#endif



static inline u_int64_t ff_getCpuFreq() {
    u_int64_t  t = 0;
#if defined(__linux__)
    FILE       *f;
    float       mhz;

    f = popen("cat /proc/cpuinfo |grep MHz |head -1|sed 's/^.*: //'", "r");
    if (fscanf(f, "%f", &mhz) == EOF) {pclose(f); return t;}
    t = (unsigned long)(mhz * 1000000);
    pclose(f);
#elif defined(__APPLE__)
    size_t len = 8;
    if (sysctlbyname("hw.cpufrequency", &t, &len, NULL, 0) != 0) {
        perror("sysctl");
    }
#else
#warning "Cannot detect CPU frequency" 
#endif
    return (t);
}

static inline const int ff_numCores() {
    int  n=-1;
#if defined(__linux__)
    FILE       *f;    
    f = popen("cat /proc/cpuinfo |grep processor | wc -l", "r");
    if (fscanf(f, "%d", &n) == EOF) { pclose(f); return n;}
    pclose(f);
#elif defined(__APPLE__) // BSD
	size_t len = 4;
	if (sysctlbyname("hw.ncpu", &n, &len, NULL, 0) == -1)
	   perror("sysctl");
#else
#warning "Cannot detect num. of cores."
#endif
    return n;
}

static inline int ff_mapThreadToCpu(int cpu_id, int priority_level=0) {
#if defined(__linux__) && defined(CPU_SET)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    if (sched_setaffinity(gettid(), sizeof(mask), &mask) != 0) 
        return -1;

    if (priority_level) 
        if (setpriority(PRIO_PROCESS, gettid(),priority_level) != 0) {
            perror("setpriority:");
            return -2;
        }
#elif defined(__APPLE__) // BSD
	// Mac OS does not implement direct pinning of threads onto cores.
	// Threads can be organised in affinity set. Using requested CPU
    // tag for the set. Cores under the same L2 sare not distinguished. 
    // Should e called before running the thread.
	#define CACHE_LEVELS 3
    #define L2 2
	size_t len;
	
	if (sysctlbyname("hw.cacheconfig",NULL, &len, NULL, 0) != 0) {
	  perror("sysctl");
	} else {
	  int64_t cacheconfig[len];
	  if (sysctlbyname("hw.cacheconfig", &cacheconfig[0], &len, NULL, 0) != 0)
		perror("sysctl: unable to get hw.cacheconfig");
	  else {
		/*
		for (size_t i=0;i<CACHE_LEVELS;i++)
		  std::cerr << " Cache " << i << " shared by " <<  cacheconfig[i] << " cores\n";
		*/
		struct thread_affinity_policy mypolicy;
		// Define sets taking in account pinning is performed on L2
		mypolicy.affinity_tag = cpu_id/cacheconfig[L2];
		if (
			thread_policy_set(mach_thread_self(), THREAD_AFFINITY_POLICY, (integer_t*) &mypolicy, THREAD_AFFINITY_POLICY_COUNT) != KERN_SUCCESS
			) {
		  
		  std::cerr << "Setting affinity of thread ? (" << mach_thread_self() << 
			") failed!" << std::endl;
		  return -2;
		} else {
		  std::cerr << "Sucessfully set affinity of thread (" << 
			mach_thread_self() << ") to " << cpu_id/cacheconfig[L2] << "\n";
		}
	  }
	}

    if (priority_level) 
        if (setpriority(PRIO_PROCESS, 0 /*self */,priority_level) != 0) {
            perror("setpriority:");
            return -2;
        }
#else 
#warning "CPU_SET not defined, cannot map thread to specific CPU"
#endif
    return 0;
}
#endif
