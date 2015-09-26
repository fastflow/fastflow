#ifndef FF_LIBLFDS_H
#define FF_LIBLFDS_H

/* 
 * This file is taken from liblfds.6 a portable, license-free, 
 * lock-free data structure library written in C.
 *
 * Please visit:   http://www.liblfds.org
 * 
 */

/* 
   Modified by: Marco Aldinucci
   - Sept 2015
*/



// ------------------------------ liblfds.h file --------------------------------

/***** library header *****/
#define LIBLFDS_RELEASE_NUMBER 6



/***** abstraction *****/

/***** defines *****/
#if (defined _WIN64 && defined _MSC_VER && !defined WIN_KERNEL_BUILD)
// TRD : 64-bit Windows user-mode with the Microsoft C compiler, any CPU
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <intrin.h>
typedef unsigned __int64      atom_t;
//#define INLINE                extern __forceinline
#define ALIGN_TO_PRE(alignment)      __declspec( align(alignment) )
#define ALIGN_TO_POST(alignment) 
#define FF_MEM_ALIGN(__A,__alignment)  __declspec( align(__alignment) ) __A  
#define ALIGN_SINGLE_POINTER  8
#define ALIGN_DOUBLE_POINTER  16
#endif

#if (!defined _WIN64 && defined _WIN32 && defined _MSC_VER && !defined WIN_KERNEL_BUILD)
// TRD : 32-bit Windows user-mode with the Microsoft C compiler, any CPU
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <intrin.h>
typedef unsigned long int     atom_t;
//#define INLINE                extern __forceinline
#define ALIGN_TO_PRE(alignment)      __declspec( align(alignment) )
#define ALIGN_TO_POST(alignment)
#define FF_MEM_ALIGN(__A,__alignment)  __declspec( align(__alignment) ) __A
#define ALIGN_SINGLE_POINTER  4
#define ALIGN_DOUBLE_POINTER  8


// TRD : this define is documented but missing in Microsoft Platform SDK v7.0
#define _InterlockedCompareExchangePointer(destination, exchange, compare) _InterlockedCompareExchange((volatile long *) destination, (long) exchange, (long) compare)
#endif

#if (defined _WIN64 && defined _MSC_VER && defined WIN_KERNEL_BUILD)
// TRD : 64-bit Windows kernel with the Microsoft C compiler, any CPU
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <wdm.h>
typedef unsigned __int64      atom_t;
#define INLINE                extern __forceinline
#define FF_MEM_ALIGN(__A,__alignment)  __declspec( align(__alignment) ) __A
#define ALIGN_TO_PRE(alignment)      __declspec( align(alignment) )
#define ALIGN_TO_POST(alignment)
#define ALIGN_SINGLE_POINTER  8
#define ALIGN_DOUBLE_POINTER  16
#endif

#if (!defined _WIN64 && defined _WIN32 && defined _MSC_VER && defined WIN_KERNEL_BUILD)
// TRD : 32-bit Windows kernel with the Microsoft C compiler, any CPU
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <wdm.h>
typedef unsigned long int     atom_t;
#define INLINE                extern __forceinline
#define ALIGN_TO_PRE(alignment)      __declspec( align(alignment) )
#define ALIGN_TO_POST(alignment)
#define FF_MEM_ALIGN(__A,__alignment)  __declspec( align(__alignment) ) __A
#define ALIGN_SINGLE_POINTER  4
#define ALIGN_DOUBLE_POINTER  8

// TRD : this define is documented but missing in Microsoft Platform SDK v7.0
#define _InterlockedCompareExchangePointer(destination, exchange, compare) _InterlockedCompareExchange((volatile long *) destination, (long) exchange, (long) compare)
#endif

#if (defined __unix__ && defined __x86_64__ && __GNUC__)
// TRD : any UNIX with GCC on x64
#if !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 600
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
typedef unsigned long long int  atom_t;
#define INLINE                  inline
#define ALIGN_TO_PRE(alignment)
#define ALIGN_TO_POST(alignment)        __attribute__( (aligned(alignment)) )
#define FF_MEM_ALIGN(__A,__alignment)  __A __attribute__( (aligned(alignment)) )
#define ALIGN_SINGLE_POINTER    8
#define ALIGN_DOUBLE_POINTER    16
//#endif
#elif (defined __unix__ && (defined __i686__ || defined __i386__) && __GNUC__)
// TRD : any UNIX with GCC on x86
#if !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 600
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
typedef unsigned long int     atom_t;
#define INLINE                inline
#define ALIGN_TO_PRE(alignment)
#define ALIGN_TO_POST(alignment)      __attribute__( (aligned(alignment)) )
#define FF_MEM_ALIGN(__A,__alignment)  __A __attribute__( (aligned(alignment)) )
#define ALIGN_SINGLE_POINTER  4
#define ALIGN_DOUBLE_POINTER  8
//#endif
#elif (defined __unix__ && defined __arm__ && __GNUC__)
// TRD : any UNIX with GCC on ARM
#if !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 600
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
typedef unsigned long int     atom_t;
#define INLINE                inline
#define ALIGN_TO_PRE(alignment)
#define ALIGN_TO_POST(alignment)      __attribute__( (aligned(alignment)) )
#define FF_MEM_ALIGN(__A,__alignment)  __A __attribute__( (aligned(alignment)) )
#define ALIGN_SINGLE_POINTER  4
#define ALIGN_DOUBLE_POINTER  8
#endif


// 2011 - Marco Aldinucci - aldinuc@di.unito.it
#if (defined __APPLE__ && __GNUC__)
// TRD : any UNIX with GCC 
#if !defined(_XOPEN_SOURCE)
//#define _XOPEN_SOURCE 600
#define _XOPEN_SOURCE 700
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#if (defined __x86_64__)
typedef unsigned long long int  atom_t;
#define INLINE                  inline
#define ALIGN_TO_PRE(alignment)
#define ALIGN_TO_POST(alignment)        __attribute__( (aligned(alignment)) )
#define FF_MEM_ALIGN(__A,__alignment)  __A __attribute__( (aligned(alignment)) )
#define ALIGN_SINGLE_POINTER    8
#define ALIGN_DOUBLE_POINTER    16
#else
typedef unsigned long int        atom_t;
#define INLINE                  inline
#define ALIGN_TO_PRE(alignment)
#define ALIGN_TO_POST(alignment)        __attribute__( (aligned(alignment)) )
#define FF_MEM_ALIGN(__A,__alignment)  __A __attribute__( (aligned(alignment)) )
#define ALIGN_SINGLE_POINTER    4 
#define ALIGN_DOUBLE_POINTER    8
#endif
#endif

#endif //LIBLFDS

