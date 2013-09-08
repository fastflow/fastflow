#ifndef __LIBLFDS_H_
#define __LIBLFDS_H_

/* 
 * This file is taken from liblfds.6 a portable, license-free, 
 * lock-free data structure library written in C.
 *
 * Please visit:   http://www.liblfds.org
 * 
 *
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
#define _XOPEN_SOURCE 600
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



// ------------------------------ abstraction_dcas.c file --------------------------------


/****************************************************************************/
#if (defined _WIN64 && defined _MSC_VER)

  /* TRD : 64 bit Windows (user-mode or kernel) on any CPU with the Microsoft C compiler

           _WIN64    indicates 64 bit Windows
           _MSC_VER  indicates Microsoft C compiler
  */

  INLINE unsigned char abstraction_dcas( volatile atom_t *destination, atom_t *exchange, atom_t *compare )
  {
    unsigned char
      cas_result;

    assert( destination != NULL );
    assert( exchange != NULL );
    assert( compare != NULL );

    cas_result = _InterlockedCompareExchange128( (volatile __int64 *) destination, (__int64) *(exchange+1), (__int64) *exchange, (__int64 *) compare );

    return( cas_result );
  }

#endif





/****************************************************************************/
#if (!defined _WIN64 && defined _WIN32 && defined _MSC_VER)

  /* TRD : 32 bit Windows (user-mode or kernel) on any CPU with the Microsoft C compiler

           (!defined _WIN64 && defined _WIN32)  indicates 32 bit Windows
           _MSC_VER                             indicates Microsoft C compiler
  */

  INLINE unsigned char abstraction_dcas( volatile atom_t *destination, atom_t *exchange, atom_t *compare )
  {
    __int64
      original_compare;

    assert( destination != NULL );
    assert( exchange != NULL );
    assert( compare != NULL );

    *(__int64 *) &original_compare = *(__int64 *) compare;

    *(__int64 *) compare = _InterlockedCompareExchange64( (volatile __int64 *) destination, *(__int64 *) exchange, *(__int64 *) compare );

    return( (unsigned char) (*(__int64 *) compare == *(__int64 *) &original_compare) );
  }

#endif





/****************************************************************************/
#if (defined __x86_64__ && __GNUC__ && !defined __pic__)

  /* TRD : any OS on x64 with GCC for statically linked code

           __x86_64__  indicates x64
           __GNUC__    indicates GCC
  */

  INLINE unsigned char abstraction_dcas( volatile atom_t *destination, atom_t *exchange, atom_t *compare )
  {
    unsigned char
      cas_result;

    assert( destination != NULL );
    assert( exchange != NULL );
    assert( compare != NULL );

    __asm__ __volatile__
    (
      "lock;"           // make cmpxchg16b atomic
      "cmpxchg16b %0;"  // cmpxchg16b sets ZF on success
      "setz       %3;"  // if ZF set, set cas_result to 1

      // output
      : "+m" (*(volatile atom_t (*)[2]) destination), "+a" (*compare), "+d" (*(compare+1)), "=q" (cas_result)

      // input
      : "b" (*exchange), "c" (*(exchange+1))

      // clobbered
      : "cc", "memory"
    );

    return( cas_result );
  }

//#endif

/****************************************************************************/
#elif (defined __i686__ && __GNUC__ && !defined __pic__)

  /* TRD : any OS on x86 with GCC for statically linked code

           __i686__  indicates x86
           __GNUC__  indicates GCC
  */

  INLINE unsigned char abstraction_dcas( volatile atom_t *destination, atom_t *exchange, atom_t *compare )
  {
    unsigned char
      cas_result;

    assert( destination != NULL );
    assert( exchange != NULL );
    assert( compare != NULL );

    __asm__ __volatile__
    (
      "lock;"          // make cmpxchg8b atomic
      "cmpxchg8b %0;"  // cmpxchg8b sets ZF on success
      "setz      %3;"  // if ZF set, set cas_result to 1

      // output
      : "+m" (*(volatile atom_t (*)[2]) destination), "+a" (*compare), "+d" (*(compare+1)), "=q" (cas_result)

      // input
      : "b" (*exchange), "c" (*(exchange+1))

      // clobbered
      : "cc", "memory"
    );

    return( cas_result );
  }

//#endif


/****************************************************************************/
#elif (defined __x86_64__ && __GNUC__ && defined __pic__)

  /* TRD : any OS on x64 with GCC for position independent code (e.g. a shared object)

           __x86_64__  indicates x64
           __GNUC__    indicates GCC
  */

  INLINE unsigned char abstraction_dcas( volatile atom_t *destination, atom_t *exchange, atom_t *compare )
  {
    unsigned char
      cas_result;

    assert( destination != NULL );
    assert( exchange != NULL );
    assert( compare != NULL );

    /* TRD : with a shared object, we cannot clobber RBX
             as such, we borrow RSI - we load half of the exchange value into it
             then swap it with RBX
             then do the compare-and-swap
             then swap the original value of RBX back from RSI
    */

    __asm__ __volatile__
    (
      "xchg %%rsi, %%rbx;"  // swap RBI and RBX 
      "lock;"               // make cmpxchg16b atomic
      "cmpxchg16b %0;"      // cmpxchg16b sets ZF on success
      "setz       %3;"      // if ZF set, set cas_result to 1
      "xchg %%rbx, %%rsi;"  // re-swap RBI and RBX

      // output
      : "+m" (*(volatile atom_t (*)[2]) destination), "+a" (*compare), "+d" (*(compare+1)), "=q" (cas_result)

      // input
      : "S" (*exchange), "c" (*(exchange+1))

      // clobbered
      : "cc", "memory"
    );

    return( cas_result );
  }

//#endif


/****************************************************************************/
#elif (defined __i686__ && __GNUC__ && defined __pic__)

  /* TRD : any OS on x86 with GCC for position independent code (e.g. a shared object)

           __i686__  indicates x86
           __GNUC__  indicates GCC
  */

  INLINE unsigned char abstraction_dcas( volatile atom_t *destination, atom_t *exchange, atom_t *compare )
  {
    unsigned char
      cas_result;

    assert( destination != NULL );
    assert( exchange != NULL );
    assert( compare != NULL );

    /* TRD : with a shared object, we cannot clobber EBX
             as such, we borrow ESI - we load half of the exchange value into it
             then swap it with EBX
             then do the compare-and-swap
             then swap the original value of EBX back from ESI
    */

    __asm__ __volatile__
    (
      "xchg %%esi, %%ebx;"  // swap EBI and EBX
      "lock;"               // make cmpxchg8b atomic
      "cmpxchg8b %0;"       // cmpxchg8b sets ZF on success
      "setz      %3;"       // if ZF set, set cas_result to 1
      "xchg %%ebx, %%esi;"  // re-swap EBI and EBX

      // output
      : "+m" (*(volatile atom_t (*)[2]) destination), "+a" (*compare), "+d" (*(compare+1)), "=q" (cas_result)

      // input
      : "S" (*exchange), "c" (*(exchange+1))

      // clobbered
      : "cc", "memory"
    );

    return( cas_result );
  }

//#endif


/****************************************************************************/
#elif ((defined __i686__ || defined __arm__) && __GNUC__ >= 4 && __GNUC_MINOR__ >= 1 && __GNUC_PATCHLEVEL__ >= 0)

  /* TRD : any OS on x86 or ARM with GCC 4.1.0 or better

           GCC 4.1.0 introduced the __sync_*() atomic intrinsics

           __GNUC__ / __GNUC_MINOR__ / __GNUC_PATCHLEVEL__  indicates GCC and which version
  */

static INLINE unsigned char abstraction_dcas( volatile atom_t *destination, atom_t *exchange, atom_t *compare )
  {
    unsigned char
      cas_result = 0;

    unsigned long long int
      original_destination;

    assert( destination != NULL );
    assert( exchange != NULL );
    assert( compare != NULL );

    __asm__ __volatile__ ( "" : : : "memory" );
    original_destination = __sync_val_compare_and_swap( (volatile unsigned long long int *) destination, *(unsigned long long int *) compare, *(unsigned long long int *) exchange );
    __asm__ __volatile__ ( "" : : : "memory" );

    if( original_destination == *(unsigned long long int *) compare )
      cas_result = 1;

    *(unsigned long long int *) compare = original_destination;

    return( cas_result );
  }

#endif

// ------------------------------ abstraction_cas.c file --------------------------------

/****************************************************************************/
#if (defined _WIN32 && defined _MSC_VER)

  /* TRD : 64 bit and 32 bit Windows (user-mode or kernel) on any CPU with the Microsoft C compiler

           _WIN32    indicates 64-bit or 32-bit Windows
           _MSC_VER  indicates Microsoft C compiler
  */

  INLINE atom_t abstraction_cas( volatile atom_t *destination, atom_t exchange, atom_t compare )
  {
    assert( destination != NULL );
    // TRD : exchange can be any value in its range
    // TRD : compare can be any value in its range

    return( (atom_t) _InterlockedCompareExchangePointer((void * volatile *) destination, (void *) exchange, (void *) compare) );
  }
#define _ACAS_
#endif





/****************************************************************************/
#if (!defined __arm__ && __GNUC__ >= 4 && __GNUC_MINOR__ >= 1 && __GNUC_PATCHLEVEL__ >= 0) && !defined(_ACAS_)

  /* TRD : any OS on any CPU except ARM with GCC 4.1.0 or better

           GCC 4.1.0 introduced the __sync_*() atomic intrinsics

           __GNUC__ / __GNUC_MINOR__ / __GNUC_PATCHLEVEL__  indicates GCC and which version
  */

  inline atom_t abstraction_cas( volatile atom_t *destination, atom_t exchange, atom_t compare )
  {
    assert( destination != NULL );
    // TRD : exchange can be any value in its range
    // TRD : compare can be any value in its range
    // TRD : note the different argument order for the GCC instrinsic to the MSVC instrinsic
	//std::cerr << "CAS(dest,comp,exch) " << destination << " " << compare << " " << exchange << "\n\n";
	//std::cerr.flush();
    return( (atom_t) __sync_val_compare_and_swap(destination, compare, exchange) );
  }

#endif


/****************************************************************************/
#if (defined __arm__ && __GNUC__ >= 4 && __GNUC_MINOR__ >= 1 && __GNUC_PATCHLEVEL__ >= 0)
static INLINE atom_t abstraction_cas( volatile atom_t *destination, atom_t exchange, atom_t compare )
  {
    atom_t
      rv;

    assert( destination != NULL );
    // TRD : exchange can be any value in its range
    // TRD : compare can be any value in its range

    // TRD : note the different argument order for the GCC instrinsic to the MSVC instrinsic


    __asm__ __volatile__ ( "" : : : "memory" );
    rv = (atom_t) __sync_val_compare_and_swap( destination, compare, exchange );
    __asm__ __volatile__ ( "" : : : "memory" );

    return( rv );
  }
#endif

#endif // __LIBLFDS_H_
