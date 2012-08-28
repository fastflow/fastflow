/** \file fasta.h
 *
 * Routines for reading FASTA databases.
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

#ifndef FASTA_H
#define FASTA_H

#include "swps3.h"
#include <stdio.h>

typedef struct{
	FILE *fp;
	char *name;
	char *data;
	char readBuffer[ MAX_SEQ_LENGTH*10 + 15 ]  __ALIGNED__;
}FastaLib;

FastaLib * swps3_openLib( char * filename );
char * swps3_readNextSequence( FastaLib * lib, int * len );
char * swps3_getSequenceName( FastaLib * lib );
void swps3_closeLib( FastaLib * lib );
void swps3_translateSequence(char *sequence, int seqLen, char table[256]);

#if defined(FAST_FLOW)
#include <fastflow.h>   // define task_t
#include <ff/allocator.hpp>

typedef struct {
    FILE *fp;
#if defined(FF_ALLOCATOR)
    ff::ff_allocator * allocator;
#else
    void * allocator;
#endif
} FastaLib_ff;
FastaLib_ff * ff_openLib( char * filename, ALLOCATOR_T * allocator);

char *        ff_readNextSequence( FastaLib_ff * lib, task_t *& task );

void          ff_closeLib( FastaLib_ff * lib );

#endif /* FAST_FLOW */

#endif /* FASTA_H */
