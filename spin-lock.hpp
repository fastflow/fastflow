/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_SPINLOCK_HPP_
#define _FF_SPINLOCK_HPP_
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

#include <sysdep.h>

namespace ff {

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


} // namespace ff

#endif /* _FF_SPINLOCK_HPP_ */
