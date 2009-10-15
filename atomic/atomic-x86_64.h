/* Massimo: 
 *  This is a slightly modified version of the linux kernel file 
 *   /<source-dir>/include/asm-x86/atomic_64.h
 *
 */
#ifndef __ARCH_X86_64_ATOMIC__
#define __ARCH_X86_64_ATOMIC__


/* atomic_t should be 32 bit signed type */

/*
 * Atomic operations that C can't guarantee us.  Useful for
 * resource counting etc..
 */

#define LOCK_PREFIX "lock ; "


/*
 * Make sure gcc doesn't try to be clever and move things around
 * on us. We need to use _exactly_ the address the user gave us,
 * not some alias that contains the same information.
 */
typedef struct { volatile int counter; } atomic_t;

#define ATOMIC_INIT(i)	{ (i) }

/**
 * atomic_read - read atomic variable
 * @v: pointer of type atomic_t
 * 
 * Atomically reads the value of @v.
 */ 
#define atomic_read(v)		((v)->counter)

/**
 * atomic_set - set atomic variable
 * @v: pointer of type atomic_t
 * @i: required value
 * 
 * Atomically sets the value of @v to @i.
 */ 
#define atomic_set(v,i)		(((v)->counter) = (i))


/**
 * atomic_inc - increment atomic variable
 * @v: pointer of type atomic_t
 * 
 * Atomically increments @v by 1.
 */ 
static __inline__ void atomic_inc(atomic_t *v)
{
	__asm__ __volatile__(
		LOCK_PREFIX "incl %0"
		:"=m" (v->counter)
		:"m" (v->counter));
}

#define atomic_dec_return(v)  (atomic_sub_return(1,v))

/* An 64bit atomic type */

typedef struct { volatile long counter; } atomic64_t;

#define ATOMIC64_INIT(i)	{ (i) }

/**
 * atomic64_read - read atomic64 variable
 * @v: pointer of type atomic64_t
 *
 * Atomically reads the value of @v.
 * Doesn't imply a read memory barrier.
 */
#define atomic64_read(v)		((v)->counter)

/**
 * atomic64_set - set atomic64 variable
 * @v: pointer to type atomic64_t
 * @i: required value
 *
 * Atomically sets the value of @v to @i.
 */
#define atomic64_set(v,i)		(((v)->counter) = (i))


/**
 * atomic64_inc - increment atomic64 variable
 * @v: pointer to type atomic64_t
 *
 * Atomically increments @v by 1.
 */
static __inline__ void atomic64_inc(atomic64_t *v)
{
	__asm__ __volatile__(
		LOCK_PREFIX "incq %0"
		:"=m" (v->counter)
		:"m" (v->counter));
}
#endif
