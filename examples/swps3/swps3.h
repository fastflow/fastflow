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

#ifndef SWPS3_H
#define SWPS3_H

/* For DMA reasons this value needs to be a multiple of 16!!! */
#define MAX_SEQ_LENGTH (1024*64)
#define MAX_SEQ_NAME_LENGTH (255)
#define MATRIX_DIM ('Z'-'A'+1)


#ifdef __GNUC__
#define __ALIGNED__ __attribute__((__aligned__(16)))
#define LIKELY(x) __builtin_expect((x),1) /** static branch prediction */
#define UNLIKELY(x) __builtin_expect((x),0) /** static branch prediction */
#define EXPORT __attribute__((visibility("default"))) /** symbol should be exported from DSO */
#else
#define __ALIGNED__
#define LIKELY(x) (x) /** static branch prediction */
#define UNLIKELY(x) (x) /** static branch prediction */
#define EXPORT /** symbol should be exported from DSO */
#endif

#if (defined(_MSC_VER) && defined(_WIN32))
#define __WIN_ALIGNED_16__ __declspec(align(16))
#else
#define __WIN_ALIGNED_16__ 
#endif

typedef enum{
	SCALAR,
	SSE2,
	PS3,
	ALTIVEC
} SWType;

typedef struct{
	double gapOpen /** gap initiation penalty */;
	double gapExt /** gap extension penalty */;
	double threshold /** maximum score */;
} Options;

#endif /* SWPS3_H */
