/** \file DynProgr_sse_short.h
 *
 * Profile generation and alignment for packed short vectors on SSE2.
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

#ifndef STRIPED_SHORT_H
#define STRIPED_SHORT_H

#include "swps3.h"
#include "matrix.h"
#include <xmmintrin.h>

typedef struct{
	int len;
	unsigned char bias;
	__m64   * profile;
	__m128i * rD;
	__m128i * storeOpt;
	__m128i * loadOpt;
	char data[1];
} ProfileShort;

ProfileShort * swps3_createProfileShortSSE( const char * query, int queryLen, SBMatrix matrix );
double swps3_alignmentShortSSE( ProfileShort * query, const char * db, int dbLen, Options * options );
void swps3_freeProfileShortSSE( ProfileShort * profile );

#endif /* STRIPED_SHORT_H */
