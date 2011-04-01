/*  FastForward queue remix
 *  Copyright (c) 2011, Dmitry Vyukov
 *  Distributed under the terms of the GNU General Public License as published by the Free Software Foundation,
 *  either version 3 of the License, or (at your option) any later version.
 *  See: http://www.gnu.org/licenses
 */ 

#pragma once
#include <windows.h>
#include <process.h>
#include <intrin.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>

#define INLINE __forceinline
#define NOINLINE __declspec(noinline)
#define CACHE_LINE_SIZE 128


INLINE void WMB() {}

INLINE void* aligned_malloc(size_t sz)
{
    return _aligned_malloc(sz, CACHE_LINE_SIZE);
}


INLINE void aligned_free(void* mem)
{
    _aligned_free(mem);
}


INLINE int get_proc_count()
{
    SYSTEM_INFO         info;
    GetSystemInfo(&info);
    return info.dwNumberOfProcessors;
}


INLINE long long get_nano_time(int start)
{
    LARGE_INTEGER time;
    LARGE_INTEGER fr;

    if (start)
    {
        QueryPerformanceFrequency(&fr);
        QueryPerformanceCounter(&time);
    }
    else
    {
        QueryPerformanceCounter(&time);
        QueryPerformanceFrequency(&fr);
    }

    return time.QuadPart * 1000000000 / fr.QuadPart;
}


INLINE void sleep_ms(int msec)
{
    Sleep(msec);
}


INLINE void atomic_signal_fence()
{
    _ReadWriteBarrier();
}


INLINE void atomic_thread_fence()
{
    _mm_mfence();
}


INLINE int atomic_add_fetch(int volatile* addr, int val)
{
    return _InterlockedExchangeAdd((long volatile*)addr, val) + val;
}


INLINE void atomic_addr_store_release(void* volatile* addr, void* val)
{
    _ReadWriteBarrier();
    addr[0] = val;
}


INLINE void* atomic_addr_load_acquire(void* volatile* addr)
{
    void* val;
    val = addr[0];
    _ReadWriteBarrier();
    return val;
}

INLINE uint64_t atomic_uint64_compare_exchange(uint64_t volatile* addr, uint64_t cmp, uint64_t xchg)
{
    return (uint64_t)_InterlockedCompareExchange64((__int64 volatile*)addr, xchg, cmp);
}

INLINE void* atomic_addr_compare_exchange(void* volatile* addr, void* cmp, void* xchg)
{
#ifdef _M_X64
        return (void*)_InterlockedCompareExchange64((__int64 volatile*)addr, (__int64)xchg, (__int64)cmp);
#else
        return (void*)_InterlockedCompareExchange((long volatile*)addr, (long)xchg, (long)cmp);
#endif
}

INLINE unsigned atomic_uint_fetch_add(unsigned volatile* addr, unsigned val)
{
    return _InterlockedExchangeAdd((long volatile*)addr, val);
}

INLINE unsigned atomic_uint_exchange(unsigned volatile* addr, unsigned val)
{
    return (unsigned)_InterlockedExchange((long volatile*)addr, val);
}

INLINE void atomic_uint_store_release(unsigned volatile* addr, unsigned val)
{
    *addr = val;
}

INLINE unsigned atomic_uint_load_acquire(unsigned volatile* addr)
{
    return *addr;
}

INLINE void core_yield()
{
    _mm_pause();
}

INLINE void thread_yield()
{
    SwitchToThread();
}

typedef HANDLE thread_t;


INLINE void thread_start(thread_t* th, void*(*func)(void*), void* arg)
{
    th[0] = (HANDLE)_beginthreadex(0, 0, (unsigned(__stdcall*)(void*))func, arg, 0, 0);
}


INLINE void thread_join(thread_t* th)
{
    WaitForSingleObject(th[0], INFINITE);
    CloseHandle(th[0]);
}


INLINE void thread_setup_prio()
{
    //SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
}


typedef HANDLE semaphore_t;


INLINE void semaphore_create(semaphore_t* sem)
{
    sem[0] = CreateSemaphore(0, 0, INT_MAX, 0);
}


INLINE void semaphore_destroy(semaphore_t* sem)
{
    CloseHandle(sem[0]);
}


INLINE void semaphore_post(semaphore_t* sem)
{
    ReleaseSemaphore(sem[0], 1, 0);
}


INLINE void semaphore_wait(semaphore_t* sem)
{
    WaitForSingleObject(sem[0], INFINITE);
}


typedef CRITICAL_SECTION mutex_t;

INLINE void mutex_create(mutex_t* mtx)
{
    InitializeCriticalSectionAndSpinCount(mtx, 2000);
}

INLINE void mutex_destroy(mutex_t* mtx)
{
    DeleteCriticalSection(mtx);
}

INLINE void mutex_lock(mutex_t* mtx)
{
    EnterCriticalSection(mtx);
}

INLINE void mutex_unlock(mutex_t* mtx)
{
    LeaveCriticalSection(mtx);
}



typedef DWORD tls_slot_t;

INLINE void tls_create(tls_slot_t* slot)
{
    *slot = TlsAlloc();
}

INLINE void tls_destroy(tls_slot_t* slot)
{
     TlsFree(*slot);
}

INLINE void* tls_get(tls_slot_t* slot)
{
    return TlsGetValue(*slot);
}

INLINE void tls_set(tls_slot_t* slot, void* value)
{
    TlsSetValue(*slot, value);
}

// MA

#define atomic_long_read atomic_uint_load_acquire
#define atomic_long_set atomic_uint_store_release
#define atomic_long_t unsigned volatile 