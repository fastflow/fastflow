/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_SPINLOCK_HPP_
#define _FF_SPINLOCK_HPP_
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

#include <ff/sysdep.h>

#ifdef __cplusplus
namespace ff {
#define _INLINE static inline
#else
#define _INLINE __forceinline
#endif

#if defined(USE_TICKETLOCK)

// NOTE: ticket lock implementation is experimental and for Linux only!

#if !defined(__linux__)
#error "Ticket-lock implementation only for Linux!"
#endif


#if !defined(LOCK_PREFIX)
#define LOCK_PREFIX "lock ; "
#endif

typedef  struct {unsigned int slock;}  lock_t[1];
enum { UNLOCKED=0 };

static inline void init_unlocked(lock_t l) { l[0].slock=UNLOCKED;}


/* Ticket-Lock from Linux kernel 2.6.37 */
static __always_inline void spin_lock(lock_t lock)
{
	int inc = 0x00010000;
	int tmp;

	asm volatile(LOCK_PREFIX "xaddl %0, %1\n"
		     "movzwl %w0, %2\n\t"
		     "shrl $16, %0\n\t"
		     "1:\t"
		     "cmpl %0, %2\n\t"
		     "je 2f\n\t"
		     "rep ; nop\n\t"
		     "movzwl %1, %2\n\t"
		     /* don't need lfence here, because loads are in-order */
		     "jmp 1b\n"
		     "2:"
		     : "+r" (inc), "+m" (lock->slock), "=&r" (tmp)
		     :
		     : "memory", "cc");
}

static __always_inline void spin_unlock(lock_t lock)
{
	asm volatile(LOCK_PREFIX "incw %0"
		     : "+m" (lock->slock)
		     :
		     : "memory", "cc");
}

#if 0
/* The following, courtesy of Fabrizio Petrini <fpetrin@us.ibm.com> */
    
union ticketlock
{
    uint32_t u;
    struct
    {
        uint16_t ticket;
        uint16_t users;
    } s;
};

/* simple interface to the gcc atomics */
#define atomic_xadd(P, V)       __sync_fetch_and_add((P), (V))

/* pause instruction to prevent excess processor bus usage */ 
#define cpu_relax() asm volatile("pause\n": : :"memory")

typedef volatile ticketlock lock_t[1];
enum { UNLOCKED=0 };

static inline void init_unlocked(lock_t l) { l[0].u=UNLOCKED;}

static inline void spin_lock( lock_t l ) {
    uint16_t me = atomic_xadd( &(l[0].s.users), 1 );
    
    while ( l[0].s.ticket != me ) ; //cpu_relax();
}
    
static inline void spin_unlock( lock_t l ) {
    WMB();
    l[0].s.ticket++;
}
#endif


#else /* xchg-based spin-lock */
    

typedef volatile int lock_t[1];
enum { UNLOCKED=0 };

_INLINE void init_unlocked(lock_t l) { l[0]=UNLOCKED;}
_INLINE void init_locked(lock_t l)   { l[0]=!UNLOCKED;}

/*
 * NOTE: 
 *  The following 2 functions has been taken from Cilk (version 5.4.6). 
 *  The Cilk Project web site is  http://supertech.csail.mit.edu/cilk/
 *
 */

#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && defined(_WIN32)
#include <WinBase.h>
// windows platform
_INLINE void spin_lock(lock_t l) {
	while (InterlockedExchange((long *)l, 1) != UNLOCKED) {	
        while (l[0]) ;  /* spin using only reads - reduces bus traffic */
    }
}

_INLINE void spin_unlock(lock_t l) {
    /*
     *  Ensure that all previous writes are globally visible before any
     *  future writes become visible.
     */    
    WMB();
    l[0]=UNLOCKED;
}
#else // non windows platform
_INLINE void spin_lock(lock_t l) {
    while (xchg((int *)l, 1) != UNLOCKED) {
        while (l[0]) ;  /* spin using only reads - reduces bus traffic */
    }
}

_INLINE void spin_unlock(lock_t l) {
    /*
     *  Ensure that all previous writes are globally visible before any
     *  future writes become visible.
     */    
    WMB();
    l[0]=UNLOCKED;
}

#endif // windows platform spin_lock
#endif

#ifdef __cplusplus
} // namespace ff
#endif

#endif /* _FF_SPINLOCK_HPP_ */
