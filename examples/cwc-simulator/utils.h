/*
  This file is part of CWC Simulator.

  CWC Simulator is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  CWC Simulator is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with CWC Simulator.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _CWC_UTILS_H_
#define _CWC_UTILS_H_

#ifdef FF_ALLOCATOR
#include <ff/allocator.hpp>
#define ALLOCATOR_INIT()
#define MALLOC(size) (ff::FFAllocator::instance()->malloc(size))
#define FREE(ptr) (ff::FFAllocator::instance()->free(ptr))
#else
#include <stdlib.h>
#define ALLOCATOR_INIT() 
#define MALLOC(size) malloc(size)
#define FREE(ptr) free(ptr)
#endif

#define CACHE_LINE 64
//#define MYNEW(ptr, type, ...) {type *sp; type **spp = &sp; posix_memalign((void **)spp, CACHE_LINE, sizeof(type)); ptr = new (sp) type(__VA_ARGS__);}
#define MYNEW(ptr, type, ...) {ptr = new type(__VA_ARGS__);}
#include <vector>

namespace cwc_utils {
  int my_round (double);

  std::vector<int> linspace_int(int, int);

  template <class T>
    inline bool found(T item, std::vector<T> v) {
    for(unsigned int i=0; i<v.size(); ++i)
      if(v[i] == item)
	return true;
    return false;
  }
}



#ifdef __linux
#define RUSAGE_WHO RUSAGE_THREAD
//#define RUSAGE_WHO RUSAGE_SELF
#include <sys/time.h>
#include <sys/resource.h>
#else
#define RUSAGE_WHO RUSAGE_SELF
#endif

#ifdef __linux
//get per-thread (if possible) elapsed time (ms) since s
inline double get_time_from(double s, struct rusage &usage) {
  getrusage(RUSAGE_WHO, &usage);
  return (double)usage.ru_utime.tv_sec * 1e+03 + (double)usage.ru_utime.tv_usec * 1e-03 - s;
}
#else
// Currently just return wall clock time - it can be implemented as follows:
//HANDLE hProcess = GetCurrentProcess();
//    FILETIME ftCreation, ftExit, ftKernel, ftUser;
//    SYSTEMTIME stKernel;
//    SYSTEMTIME stUser;

//    GetProcessTimes(hProcess, &ftCreation, &ftExit, &ftKernel, &ftUser);
//    FileTimeToSystemTime(&ftKernel, &stKernel);
//    FileTimeToSystemTime(&ftUser, &stUser);
inline double get_time_from(double s, struct rusage &usage) {
  //getrusage(RUSAGE_WHO, &usage);
	gettimeofday(&usage.ru_utime, NULL);
  return (double)usage.ru_utime.tv_sec * 1e+03 + (double)usage.ru_utime.tv_usec * 1e-03 - s;
}	
#endif

//get external elapsed time (ms) since s
inline double get_xtime_from(double s, struct timeval &time_misuration) {
  gettimeofday(&time_misuration, NULL);
  return (double)time_misuration.tv_sec * 1e+03 + (double)time_misuration.tv_usec * 1e-03 - s;
}



//thread-to-core pinning
#ifdef PINNING
//paracool
#define EMITTER_RANK -1 //0
#define SIMENGINE_RANK -1 //1
#define ALIGNMENT_RANK 31 //2
#define WINDOWS_RANK -1 //3
#define STATS_RANK -1 //4
#ifdef USE_FF_ACCEL
#define SIMFARM_EMITTER_RANK 29 //6
#define SIMFARM_COLLECTOR_RANK 30 //7
#define SIMFARM_WORKERS_RANK 0 //starting
#endif

#else //ifdef PINNING
#define EMITTER_RANK -1
#define SIMENGINE_RANK -1
#define ALIGNMENT_RANK -1
#define WINDOWS_RANK -1
#define STATS_RANK -1
#ifdef USE_FF_ACCEL
#define SIMFARM_EMITTER_RANK -1
#define SIMFARM_COLLECTOR_RANK -1
#define SIMFARM_WORKERS_RANK -1
#endif
#endif //ifdef PINNING
#endif
