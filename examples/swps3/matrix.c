/** \file matrix.c
 *
 * Routines for reading matrix files and converting matrices.
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

#include "swps3.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "debug.h"
#include <sys/types.h>
#include <math.h>

static char * skip( char * line ){
	while (isspace( * line )) line ++;
	return line;
}

static int8_t sbmatrix[MATRIX_DIM][MATRIX_DIM] __ALIGNED__;
static double dmatrix[MATRIX_DIM][MATRIX_DIM] __ALIGNED__;

EXPORT DMatrix swps3_readDMatrix( char * filename ){
	FILE * fp;
	int i;
	char line[ 1024 ];
	char topChar[ 64 ];
	int count=0;
	memset( dmatrix, 0, sizeof(dmatrix) );
	fp = fopen( filename, "r" );
	if (!fp){
		perror("Matrix - Error opening file:" );
		exit( 1 );
	}
	/* parse the first line containing the caption ( "A R N ..." )*/
	while ( !count && fgets( line, sizeof( line ), fp ) ){
		char * pline = line;
		while ( (pline = skip( pline )) ){
			/* Skip the comment and empty lines */
			if ( *pline=='#' || !*pline )
				break;
			if ( *pline<'A' || *pline>'Z'){
				error("Matrix: Invalid matrix desciption (Character expected in the caption line)" );
			}
			topChar[ count++ ] = (char)*pline-'A';
			pline++;
		}
	}
	/* Parse the left charactes and values */
	while ( fgets( line, sizeof( line ), fp ) ){
		char * pline = skip( line );
		/* Read the first character on a line */
		char leftChar = (char)*pline-'A';
		/* Skip the comment and empty lines */
		if ( *pline=='#' || !*pline ) continue;
		if ( *pline<'A' || *pline>'Z'){
			error("Matrix: Invalid matrix desciption (Character expected at the beginning of the line)" );
		}
		/* Read in the values of the matrix */
		for( i=0; i<count; i++ ){
			pline = skip( pline+1 );
			if (!isdigit( *pline ) && *pline!='-' && *pline!='.'){
				error("Matrix: Excepted a matrix value got '%c'", *pline );
			}
			dmatrix[ (int)leftChar ][ (int)topChar[ i ] ] = strtod( pline, &pline );
		}
	}
	fclose( fp );
	return (DMatrix)dmatrix;
}

EXPORT SBMatrix swps3_readSBMatrix( char * filename ){
	FILE * fp;
	int i;
	char line[ 1024 ];
	char topChar[ 64 ];
	int count=0;
	memset( sbmatrix, 0, sizeof(sbmatrix) );
	fp = fopen( filename, "r" );
	if (!fp){
		perror("Matrix - Error opening file:" );
		exit( 1 );
	}
	/* parse the first line containing the caption ( "A R N ..." )*/
	while ( !count && fgets( line, sizeof( line ), fp ) ){
		char * pline = line;
		while ( (pline = skip( pline )) ){
			/* Skip the comment and empty lines */
			if ( *pline=='#' || !*pline )
				break;
			if ( *pline<'A' || *pline>'Z'){
				error("Matrix: Invalid matrix desciption (Character expected in the caption line)" );
			}
			topChar[ count++ ] = (char)*pline-'A';
			pline++;
		}
	}
	/* Parse the left charactes and values */
	while ( fgets( line, sizeof( line ), fp ) ){
		char * pline = skip( line );
		/* Read the first character on a line */
		char leftChar = (char)*pline-'A';
		/* Skip the comment and empty lines */
		if ( *pline=='#' || !*pline ) continue;
		if ( *pline<'A' || *pline>'Z'){
			error("Matrix: Invalid matrix desciption (Character expected at the beginning of the line)" );
		}
		/* Read in the values of the matrix */
		for( i=0; i<count; i++ ){
			pline = skip( pline+1 );
			if (!isdigit( *pline ) && *pline!='-' && *pline!='.'){
				error("Matrix: Excepted a matrix value got '%c'", *pline );
			}
			sbmatrix[ (int)leftChar ][ (int)topChar[ i ] ] = strtol( pline, &pline, 10 );
		}
	}
	fclose( fp );
	return (SBMatrix)sbmatrix;
}

EXPORT SBMatrix swps3_convertMatrixD2B( double factor ) {
	int i,j;
	for (i=0; i<MATRIX_DIM; i++)  {
		for(j=0; j<MATRIX_DIM; j++) {
			double val = dmatrix[i][j]*factor;
			if ((int8_t)val < val)
				dmatrix[i][j] = (int8_t)val + 1;
			else
				dmatrix[i][j] = (int8_t)val;
		}
	}
	return (SBMatrix)sbmatrix;
}

EXPORT double swps3_factorFromThreshold( double threshold, double singleGapCost ) {
	(void)singleGapCost;
	/* return 256.0/(threshold+(max-min)+1.0);  FIXME is this right??? */
	return 256.0/threshold;
}

