#ifndef _FF_MINPORT_WIN_H_
#define _FF_MINPORT_WIN_H_
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

/* **************************************************************************
Ultra-minimal and incomplete compatibility layer for pthreads on Windows platform. Errors are not managed at all. Designed to enable a basic porting of FastFlow on Windows platform, if you plan to extensively use pthread consider to install a full pthread port for windows. Some of the primitives are inspired to Fastflow queue remix by Dmitry Vyukov http://www.1024cores.net/

March 2011 - Ver 0: Basic functional port, tested on Win 7 x64 - Performance not yet extensively tested.

*/

#pragma once

#include <windows.h>
#include <process.h>
#include <intrin.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>

#define INLINE __forceinline
#define NOINLINE __declspec(noinline)
#define RESTRICT __declspec()


// 
#define PTHREAD_CANCEL_ENABLE        0x01  /* Cancel takes place at next cancellation point */
#define PTHREAD_CANCEL_DISABLE       0x00  /* Cancel postponed */
#define PTHREAD_CANCEL_DEFERRED      0x02  /* Cancel waits until cancellation point */

// MA - pthread very partial windows port - for fastflow usage only
typedef HANDLE pthread_t;
typedef struct _opaque_pthread_attr_t { long __sig; } pthread_attr_t;
typedef struct _opaque_pthread_mutexattr_t { long __sig; } pthread_mutexattr_t;
typedef struct _opaque_pthread_condattr_t {long __sig; } pthread_condattr_t;
typedef DWORD pthread_key_t;


// Mutex and cond vars

typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;


INLINE int pthread_create(pthread_t RESTRICT * thread,
		const pthread_attr_t RESTRICT * attr, 
		void *(*start_routine)(void *), 
		void RESTRICT * arg) {
			//std::cerr << "Creating new thread\n";
			thread[0] = (HANDLE)_beginthreadex(0, 0, (unsigned(__stdcall*)(void*))start_routine, arg, 0, 0);
		return(0); 
}


INLINE int pthread_join(pthread_t thread, void **value_ptr) {
	LPDWORD exitcode = 0;
    WaitForSingleObject(thread, INFINITE);
	if (value_ptr)  {
		GetExitCodeThread(thread,exitcode);
		*value_ptr = exitcode;
	}
    CloseHandle(thread);
	return(0); // Errors are not (yet) managed.	 
}

INLINE void pthread_exit(void *value_ptr) {
	if (value_ptr)
		ExitThread(*((DWORD *) value_ptr));
	else 
		ExitThread(0);
}

INLINE int pthread_attr_init(pthread_attr_t *attr) {
	// do nothing currently
	return(0);
}

INLINE int pthread_attr_destroy(pthread_attr_t *attr) {
	// do nothing currently
	return(0);
}

INLINE int pthread_setcancelstate(int state, int *oldstate) {
	// do nothing currently
	return(0);
}

INLINE int pthread_mutex_init(pthread_mutex_t  RESTRICT * mutex,
	const pthread_mutexattr_t RESTRICT  * attr) {
	if (attr) return(EINVAL);
	InitializeCriticalSectionAndSpinCount(mutex, 1500 /*spin count */);	
	return (0); // Errors partially managed
}

INLINE int pthread_cond_init(pthread_cond_t  RESTRICT * cond,
    const pthread_condattr_t  RESTRICT * attr) {
	if (attr) return(EINVAL);
	InitializeConditionVariable(cond);	
	return (0);	 // Errors not managed
 }

INLINE int pthread_mutex_lock(pthread_mutex_t *mutex) {
	EnterCriticalSection(mutex);
	return(0); // Errors not managed
 }

INLINE int pthread_mutex_unlock(pthread_mutex_t *mutex) {
	LeaveCriticalSection(mutex);
	return(0); // Errors not managed
 }

INLINE int pthread_mutex_destroy(pthread_mutex_t *mutex) {
	DeleteCriticalSection(mutex);
	return(0); // Errors not managed
 }
 
INLINE int pthread_cond_signal(pthread_cond_t *cond) {
	WakeConditionVariable(cond);
	return(0); // Errors not managed
 }

INLINE int pthread_cond_broadcast(pthread_cond_t *cond) {
	WakeAllConditionVariable(cond);
	return(0);
 }

INLINE int pthread_cond_wait(pthread_cond_t  RESTRICT * cond,
					   pthread_mutex_t  RESTRICT * mutex) {
    SleepConditionVariableCS(cond, mutex, INFINITE);
	//WaitForSingleObject(cond[0], INFINITE); 
	return(0); // Errors not managed
 }

INLINE int pthread_cond_destroy(pthread_cond_t *cond) {
	// Do nothing. Did not find a Windows call .... 
	return (0); // Errors not managed
 }

INLINE  pthread_t pthread_self(void) {
	return(GetCurrentThread());
}

INLINE int pthread_key_create(pthread_key_t *key, void (*destructor)(void *)) {
	*key = TlsAlloc();
	return 0;
}

INLINE int pthread_key_delete(pthread_key_t key) {
	TlsFree(key);
	return 0;
}

INLINE  int pthread_setspecific(pthread_key_t key, const void *value) {
	TlsSetValue(key, (LPVOID) value);
	return (0);
}

INLINE void * pthread_getspecific(pthread_key_t key) {
	return NULL;
}

 // Other

typedef unsigned long useconds_t;
#define stoll strtoll


INLINE int usleep(unsigned long microsecs) {
	Sleep(microsecs/1000);
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

#endif // _FF_MINPORT_WIN_H_
