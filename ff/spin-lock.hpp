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


namespace ff {

#if defined(USE_TICKETLOCK)

/* Ticket-Lock, courtesy of Fabrizio Petrini <fpetrin@us.ibm.com> */

union ticketlock
{
  unsigned u;
  struct
  {
    unsigned short ticket;
    unsigned short users;
  } s;
};

/* simple interface to the gcc atomics */
#define atomic_xadd(P, V)       __sync_fetch_and_add((P), (V))

/* pause instruction to prevent excess processor bus usage */ 
#define cpu_relax() asm volatile("pause\n": : :"memory")

typedef ticketlock lock_t[1];
enum { UNLOCKED=0 };

static inline void init_unlocked(lock_t l) { l[0].u=UNLOCKED;}

static inline void spin_lock( lock_t l ) {
    unsigned short me = atomic_xadd( &(l[0].s.users), 1 );
  
    while ( l[0].s.ticket != me ) 
        cpu_relax();
}


static inline void spin_unlock( lock_t l ) {
  l[0].s.ticket++;
}

#else /* xchg-based spin-lock */
    

typedef volatile int lock_t[1];
enum { UNLOCKED=0 };

static inline void init_unlocked(lock_t l) { l[0]=UNLOCKED;}
static inline void init_locked(lock_t l)   { l[0]=!UNLOCKED;}

/*
 * NOTE: 
 *  The following 2 functions has been taken from Cilk (version 5.4.6). 
 *  The Cilk Project web site is  http://supertech.csail.mit.edu/cilk/
 *
 */
static inline void spin_lock(lock_t l) {
    while (xchg((int *)l, 1) != UNLOCKED) {
        while (l[0]) ;  /* spin using only reads - reduces bus traffic */
    }
}

static inline void spin_unlock(lock_t l) {
    /*
     *  Ensure that all previous writes are globally visible before any
     *  future writes become visible.
     */    
    WMB();
    l[0]=UNLOCKED;
}
#endif


} // namespace ff

#endif /* _FF_SPINLOCK_HPP_ */
