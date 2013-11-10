/** \file DynProgr_altivec.h
 *
 * Profile generation and alignment on IBM Altivec^TM instruction set.
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

#ifndef DYNPROGR_ALTIVEC_H_
#define DYNPROGR_ALTIVEC_H_

#include "swps3.h"

#ifdef __cplusplus
extern "C" {
#endif

double swps3_dynProgrByteAltivec(const char *db, int dbLen, void* profile, Options *options);
double swps3_dynProgrShortAltivec(const char *db, int dbLen, void* profile, Options *options);
double swps3_dynProgrFloatAltivec(const char *db, int dbLen, void* profile, Options *options);
void *swps3_createProfileByteAltivec(const char *query, int queryLen, SBMatrix matrix);
void *swps3_createProfileShortAltivec(const char *query, int queryLen, SBMatrix matrix);
void *swps3_createProfileFloatAltivec(const char *query, int queryLen, SBMatrix matrix);
  
void swps3_freeProfileByteAltivec(void *profile);
void swps3_freeProfileShortAltivec(void *profile);
void swps3_freeProfileFloatAltivec(void *profile);
#ifdef __cplusplus
}
#endif

#endif
