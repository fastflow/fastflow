/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_PLATFORM_HPP_
#define _FF_PLATFORM_HPP_
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


#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && defined(_WIN32)
#pragma unmanaged

#include "ff/platforms/pthread_minport_windows.h"
#define INLINE __forceinline
#define NOINLINE __declspec(noinline)
#define CACHE_LINE_SIZE 64

INLINE void WMB() {}

INLINE int posix_memalign(void **memptr,size_t alignment, size_t sz)
{
    *memptr =  _aligned_malloc(sz, alignment);
	return(!memptr);
}


INLINE void posix_memalign_free(void* mem)
{
    _aligned_free(mem);
}

 // Other

#include <string>
typedef unsigned long useconds_t;
//#define strtoll std::stoll
#define strtoll _strtoi64

INLINE static int usleep(unsigned long microsecs) {
  if (microsecs > 100000)
    /* At least 100 mS. Typical best resolution is ~ 15ms */
    Sleep (microsecs/ 1000);
  else
    {
      /* Use Sleep for the largest part, and busy-loop for the rest. */
      static double frequency;
      if (frequency == 0)
        {
          LARGE_INTEGER freq;
          if (!QueryPerformanceFrequency (&freq))
            {
              /* Cannot use QueryPerformanceCounter. */
              Sleep (microsecs / 1000);
              return 0;
            }
          frequency = (double) freq.QuadPart / 1000000000.0;
        }
      long long expected_counter_difference = 1000 * microsecs * (long long) frequency;
      int sleep_part = (int) (microsecs) / 1000 - 10;
      LARGE_INTEGER before;
      QueryPerformanceCounter (&before);
      long long expected_counter = before.QuadPart + 
expected_counter_difference;
      if (sleep_part > 0)
        Sleep (sleep_part);
      for (;;)
        {
          LARGE_INTEGER after;
          QueryPerformanceCounter (&after);
          if (after.QuadPart >= expected_counter)
            break;
        }
    }
	return(0);
}

//#define __TICKS2WAIT 1000
#define random rand
#define srandom srand
#define getpid _getpid
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

/*
#ifndef _TIMEVAL_DEFINED
#define _TIMEVAL_DEFINED
struct timeval {
  long tv_sec;
  long tv_usec;
};
#endif 
*/

struct timezone 
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};
 
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
  FILETIME ft;
  unsigned __int64 tmpres = 0;
  static int tzflag;
 
  if (NULL != tv)
  {
    GetSystemTimeAsFileTime(&ft);
 
    tmpres |= ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |= ft.dwLowDateTime;
 
    /*converting file time to unix epoch*/
    tmpres -= DELTA_EPOCH_IN_MICROSECS; 
    tmpres /= 10;  /*convert into microseconds*/
    tv->tv_sec = (long)(tmpres / 1000000UL);
    tv->tv_usec = (long)(tmpres % 1000000UL);
  }
 
  if (NULL != tz)
  {
    if (!tzflag)
    {
      _tzset();
      tzflag++;
    }
    tz->tz_minuteswest = _timezone / 60;
    tz->tz_dsttime = _daylight;
  }
 
  return 0;
}


//#include "ff/platforms/platform_msvc_windows.h"
//#if (!defined(_FF_SYSTEM_HAVE_WIN_PTHREAD))
//#endif
#include<algorithm>
#elif defined(__GNUC__) && (defined(__linux) || defined(__APPLE__))
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
//#define __TICKS2WAIT 1000
#else
#   error "unknown platform"
#endif

#endif //_FF_PLATFORM_HPP_


