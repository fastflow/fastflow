/** \file DynProgr_scalar.c
 *
 * Scalar alignment routine.
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

#include <sys/types.h>
#include "swps3.h"
#include "debug.h"
#include "matrix.h"

#define INFINITY (1.0/0.0)
EXPORT double swps3_alignScalar( const DMatrix matrix, const char *s1, int ls1, const char *s2, int ls2, Options *options)
{ 
    /* WARNING: this could be too big to be allocated on the stack! */
    double coldel[MAX_SEQ_LENGTH], S[MAX_SEQ_LENGTH];

    int i, j, k;
    const double DelFixed = options->gapOpen;
    const double DelIncr = options->gapExt;
    double *Score_s1;
    double Tcd, t, MaxScore = 0, Sj, Sj1, Tj, Tj1, Trd;
    
    S[0] = coldel[0] = 0;
    for( j=0; j < ls2; j++ ) {
	coldel[j] = -INFINITY;
	S[j] = 0;
    }
    
    for( i=0; i < ls1; i++ ) {
	
	Tj1 = Sj1 = 0;
	Trd = -INFINITY;
	
	/* setup Score_s1 */
	k = s1[i];
	Score_s1 = matrix+k*MATRIX_DIM;
	
        for( j=0; j < ls2; j++ ) {
            Sj = S[j];
            Tcd = coldel[j] + DelIncr;
	    
            if( Tcd < ( t=Sj+DelFixed ) ) Tcd = t;
            Tj = Sj1 + Score_s1[(int)s2[j]];
	    
            Trd += DelIncr;
            if( Trd < ( t=Tj1+DelFixed ) ) Trd = t;
	    
            if( Tj < Tcd ) Tj = Tcd;
            if( Tj < Trd ) Tj = Trd;
            if( Tj < 0 ) Tj = 0;
	    if( Tj > MaxScore ) {
		MaxScore = Tj;
		if( MaxScore >= options->threshold ) { return( MaxScore ); }
	    }
	    
            coldel[j] = Tcd;
            S[j] = Tj1 = Tj;
            Sj1 = Sj;
	}
    }
    
    return( MaxScore );
    
}
