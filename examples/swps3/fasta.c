/** \file fasta.c
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

#include "fasta.h"
#include "swps3.h"
#include "debug.h"
#include <stdlib.h>
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif /* HAVE_MALLOC_H */
#include <string.h>
#include <errno.h>

EXPORT FastaLib * swps3_openLib( char * filename ){
	FastaLib * lib = NULL;
	FILE * fp;
	int len;
	if ( (fp = fopen( filename, "r" )) ){
#ifdef HAVE_MALLOC_H
		lib = (FastaLib *)memalign( 16, sizeof( FastaLib ) );
#else
		lib = (FastaLib *)malloc( sizeof( FastaLib ) );
#endif /* HAVE_MALLOC_H */
		lib->fp = fp;
		lib->name = '\0';
	}
	else{
		error("Fasta: %s\n",strerror(errno));
	}
	swps3_readNextSequence( lib, &len );
	if (len)
		rewind( lib->fp );
	return lib;
}
EXPORT char * swps3_readNextSequence( FastaLib * lib, int * len ){
	char * pres = lib->readBuffer;
	if ( feof( lib->fp ) )
		return NULL;
	if( (*pres = fgetc(lib->fp)) != '>' ) {
	   warning("Missing comment line, trying to continue anyway\n");
	   lib->name = '\0';
	   lib->data = pres++;
	   goto readseq;
	} else {
           fgets( pres, sizeof(lib->readBuffer)-(pres-lib->readBuffer), lib->fp );
	   lib->name = pres;
	   for (;*pres && *pres!='\n';pres++) ;
	   *pres++ = '\0';
	   while ((long)pres&0xf) *pres++ = '\0';
	   lib->data = pres;
	}
	while ( (*pres = fgetc( lib->fp) ) != '>' ){
readseq:
                if( fgets( pres+1, sizeof(lib->readBuffer)-(pres+1-lib->readBuffer), lib->fp ) == 0 ) goto finish;
	        for (;*pres && *pres!='\n';pres++)
			if ('A'>*pres || *pres>'Z'){
				error("Invalid character in input sequence '%c'\n", *pres);
			}
	}
	*pres = '\0';
	ungetc( '>', lib->fp );
finish:
	if (len)
		*len = pres - lib->data;
	swps3_translateSequence(lib->data,pres - lib->data,NULL);
	return lib->data;
}
EXPORT char * swps3_getSequenceName( FastaLib * lib ){
        return lib->name;
}
EXPORT void swps3_closeLib( FastaLib * lib ){
	free( lib );
}
EXPORT void swps3_translateSequence(char *sequence, int seqLen, char table[256]) {
   int i;
   for(i=0;i<seqLen && sequence[i]!='\n' && sequence[i]!='\0';++i) {
      if(table) sequence[i] = table[(int)sequence[i]];
      else sequence[i] -= 'A';

      if(sequence[i] < 0 || sequence[i] >= MATRIX_DIM) error("Invalid character in input sequence at position %d\n",i);
   }
}


#if defined(FAST_FLOW)
#include <fastflow.h>
#include <ff/allocator.hpp>

EXPORT  FastaLib_ff * ff_openLib( char * filename, ALLOCATOR_T * allocator ) {
    FILE * fp;
    FastaLib_ff * lib = NULL;
    lib =(FastaLib_ff *) malloc( sizeof( FastaLib_ff ) );

    lib->allocator = allocator;

    if ( (fp = fopen( filename, "r" )) )
	lib->fp = fp;
    else
	error("Fasta: %s\n",strerror(errno));
    
    return lib;
}

#if 0
EXPORT char * ff_readNextSequenceB( FastaLib_ff * lib, char ** db, char ** data, int * len ) {
    if ( feof( lib->fp ) ) {
	if (len)  *len  = 0;
	if (db)   *db   = NULL;
	if (data) *data = NULL;
	return NULL;
    }
 
    int sz = MIN_ALLOC;
#if !defined(USE_LIBC_ALLOCATOR)
    char * _db = (char *)lib->allocator->malloc(sz);
#else
    char * _db = (char *)MALLOC(sz);
#endif
    char * pres = _db;
    char * _data;
    
    if( (*pres = fgetc(lib->fp)) != '>' ) {
	warning("Missing comment line, trying to continue anyway\n");
	_data = pres++;
	goto readseq;
    } else {
	fgets( pres, sz, lib->fp );
	for ( ; *pres && *pres!='\n'; pres++) ;
	*pres++ = '\0';
	while ((long)pres&0xf) *pres++ = '\0';
	_data = pres;
    }
    int size;
    while ( (*pres = fgetc( lib->fp) ) != '>' ){
 readseq:
	size = sz-(pres+1-_db);
	if (!size) {
	    int dataoffset=_data-_db;
#if !defined(USE_LIBC_ALLOCATOR)
	    _db = (char *)lib->allocator->realloc(_db,sz+STEP_ALLOC);
#else
	    _db = (char *)REALLOC(_db,sz+STEP_ALLOC);
#endif
	    pres = _db+sz-1;
	    _data = _db+dataoffset;
	    size = STEP_ALLOC;
	    sz   += size;
	}	    
	if( fgets( pres+1, size, lib->fp ) == 0 ) goto finish;
	for (;*pres && *pres!='\n';pres++)
	    if ('A'>*pres || *pres>'Z'){
		error("Invalid character in input sequence '%c'\n", *pres);
	    }
    }
    *pres = '\0';
    ungetc( '>', lib->fp );
 finish:
    if (len)  *len  = pres - _data;
    swps3_translateSequence(_data,pres - _data,NULL);
    if (db)   *db   = _db;
    if (data) *data = _data;
    return _db;
}
#endif

EXPORT char * ff_readNextSequence( FastaLib_ff * lib, task_t *& task) {
    if ( feof( lib->fp ) ) {
	task->dbLen = 0;
	task->dbdata = NULL;
	return NULL;
    }
    int sz = MIN_ALLOC-sizeof(task_t);
    char * pres = task->db;
    char * _data;
    
    if( (*pres = fgetc(lib->fp)) != '>' ) {
	warning("Missing comment line, trying to continue anyway\n");
	_data = pres++;
	goto readseq;
    } else {
	fgets( pres, sz, lib->fp );
	for ( ; *pres && *pres!='\n'; pres++) ;
	*pres++ = '\0';
	while ((long)pres&0xf) *pres++ = '\0';
	_data = pres;
    }

    int size;
    while ( ( (*pres = fgetc( lib->fp) ) != '>')  && ( *pres != '\n') ){
 readseq:
	size = sz-(pres+1-task->db);
	if (!size) {
	    int dataoffset=_data-task->db;
#if defined(FF_ALLOCATOR)
	    task = (task_t *)lib->allocator->realloc(task,sz+sizeof(task_t)+STEP_ALLOC);
#else
	    task = (task_t *)REALLOC(task,sz+sizeof(task_t)+STEP_ALLOC);
#endif
	    task->db = ((char *)task)+sizeof(task_t);
	    pres = task->db+sz-1;
	    _data = task->db+dataoffset;
	    size = STEP_ALLOC;
	    sz   += size;
	}	    
	if( fgets( pres+1, size, lib->fp ) == 0 ) goto finish;
	for (;*pres && *pres!='\n';pres++)
	    if ('A'>*pres || *pres>'Z'){
		error("Invalid character in input sequence '%c'\n", *pres);
	    }
    }
    if (*pres == '>') ungetc( '>', lib->fp );
    *pres = '\0';
 finish:
    task->dbLen  = pres - _data;
    swps3_translateSequence(_data,task->dbLen,NULL);
    task->dbdata = _data;
    return (char *)task;
}



EXPORT  void ff_closeLib( FastaLib_ff * lib ){
    free( lib );
}

#endif /* FAST_FLOW */
