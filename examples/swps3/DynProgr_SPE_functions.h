/** \file DynProgr_SPE_functions.h
 *
 * Profile generation and alignment on Cell/BE SPE.
 */
/*
 * Copyright (c) 2007-2008 ETH Zürich, Institute of Computational Science
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef DYNPROGR_SPE_FUNCTIONS_H
#define DYNPROGR_SPE_FUNCTIONS_H
#include <spu_mfcio.h>
#include "DynProgr_SPE.h"
#define SPU

/* Define the global data */
/**************************/
extern char * s1 /** query sequence */,
            * s2 /** database sequence */;
extern int  ls1 /** query sequence length */,
            ls2 /** database sequence length */;
extern int blockStart /** start position of current profile block */,
           blockSize /** length of current profile segment */;
extern int maxDbLen /** maximal database sequence length */;
extern double mn /** minimum score */,
              mx /** maximum score */,
              fixedDel /** gap initiation penalty */,
	      incDel /** gap extension penalty */;
extern void *simi /** similarity matrix */,
            *profile /** current profile segment */,
	    *loadOpt /** temporary storage for score column */,
	    *storeOpt /** temporary storage for score column */,
	    *rD /** temporary storage for row delection scores */,
	    *maxS /** storage for intermediate score row */,
	    *delS /** storage for intermediate column deletion scores */;
extern ppu_addr_t remote_profile /** 64-bit pointer to profile location in main memory */;

#ifdef __cplusplus
extern "C" {
#endif

typedef double(*dvf_t)(void);
typedef void(*vvf_t)(void);
extern dvf_t dynProgLocal[];
extern vvf_t createProfile[];

#ifdef __cplusplus
}
#endif
#endif
