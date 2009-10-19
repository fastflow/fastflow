/** \file DynProgr_PPU.h
 *
 * PPU frontend to the alignment routines on the SPU.
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

#ifndef DYNPROGR_PPU_H_
#define DYNPROGR_PPU_H_

#include "swps3.h"
#include "matrix.h"
#include "DynProgr_SPE.h"

double swps3_alignmentProfileSPE( const char * db, int dbLen );
double swps3_alignmentByteSPE( const SBMatrix matrix, const char * query, int queryLen, const char * db, int dbLen, Options * options );
double swps3_alignmentShortSPE( const SBMatrix matrix, const char * query, int queryLen, const char * db, int dbLen, Options * options );
double swps3_alignmentFloatSPE( const SBMatrix matrix, const char * query, int queryLen, const char * db, int dbLen, Options * options );
void swps3_loadProfileByte(SPEProfile *profile, int maxDbLen, Options *options);
void swps3_loadProfileShort(SPEProfile *profile, int maxDbLen, Options *options);
void swps3_loadProfileFloat(SPEProfile *profile, int maxDbLen, Options *options);
SPEProfile * swps3_createProfileBytePPU( const char * query, int queryLen, const SBMatrix matrix, int maxDbLen );
SPEProfile * swps3_createProfileShortPPU( const char * query, int queryLen, const SBMatrix matrix, int maxDbLen );
SPEProfile * swps3_createProfileFloatPPU( const char * query, int queryLen, const SBMatrix matrix, int maxDbLen );
void swps3_freeProfilePPU(SPEProfile *profile);

#endif /* PS3_PPU_H_ */

