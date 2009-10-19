/** \file DynProgr_sse_byte.c
 *
 * Profile generation and alignment for packed byte vectors on SSE2.
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

#include "DynProgr_sse_byte.h"
#include "debug.h"
#include <unistd.h>
#include <stdio.h>
#include <float.h>

#define PAGE_ALIGN(x) (((size_t)(x)+sysconf(_SC_PAGESIZE)-1)&~(sysconf(_SC_PAGESIZE)-1))
/**
 *  Creates a profile with unsigned 8 bit integers
 */
EXPORT ProfileByte * swps3_createProfileByteSSE( const char * query, int queryLen, SBMatrix matrix ){
	int segLen  = (queryLen+15)/16;
	int i,j,k;
	int bias = 0;
	u_int8_t * pprofile;
        ProfileByte * profile = (ProfileByte*)malloc( sizeof(ProfileByte)+segLen*(MATRIX_DIM+3)*sizeof(__m128i)+64+2*sysconf(_SC_PAGESIZE) );

	profile->loadOpt     = (__m128i*) ((size_t) (profile->data + 15) & ~(0xf)) ;
	profile->storeOpt    = profile->loadOpt + segLen;
	profile->rD          = profile->storeOpt + segLen;
	profile->profile     = (__m128i*) PAGE_ALIGN(profile->rD + segLen);

	/* Init the profile */
	profile->len = queryLen;
	/* Init the byte profile */
	for(i=0; i<MATRIX_DIM; i++)
		for(j=0; j<MATRIX_DIM; j++)
			if (bias < -matrix[ i*MATRIX_DIM+j ])
				bias = -matrix[ i*MATRIX_DIM+j ];
	pprofile = (u_int8_t*)profile->profile;

	for(i=0; i<MATRIX_DIM; i++)
		for(j=0; j<segLen; j++)
			for(k=0; k<16; k++)
				if(j+k*segLen < queryLen)
					*(pprofile++) = matrix[query[j+k*segLen]*MATRIX_DIM+i]+bias;
				else
					*(pprofile++) = bias;
	profile->bias = bias;

#ifdef DEBUG
	for(i=0; i<queryLen; ++i) debug("\t%c",query[i]+'A');
	debug("\n");
#endif
	return profile;
}


EXPORT double swps3_alignmentByteSSE( ProfileByte * query, const char * db, int dbLen, Options * options )
{

	/**********************************************************************
	* This version of the code implements the idea presented in
	*
	***********************************************************************
	* Striped Smith-Waterman speeds database searches six times over other
	* SIMD implementations
	*
	* Michael Farrar, Bioinformatics, 23(2), pp. 156-161, 2007
	**********************************************************************/

	int i, j;
	unsigned char MaxScore = 0;
	int segLength = (query->len+15)/16; /* the segment length */

	__m128i * loadOpt  = query->loadOpt;
	__m128i * storeOpt = query->storeOpt;
	__m128i * rD       = query->rD;
	__m128i * current_profile;
	__m128i * swap;

	__m128i vMinimums = _mm_set1_epi32(0);

	__m128i vDelIncr  = _mm_set1_epi8(-options->gapExt);
	__m128i vDelFixed = _mm_set1_epi8(-options->gapOpen);
	__m128i vBias     = _mm_set1_epi8(query->bias);

	__m128i vMaxScore = vMinimums;	/* vMaxScore = [0,0] */

	__m128i vStoreOpt;				/* the new optimal score */
	__m128i vRD;					/* the new row deletion score */
	__m128i vCD = vMinimums;		/* the column deletion score */
	__m128i zero = vMinimums;		/* the column deletion score */
	__m128i vTmp;
#ifdef DEBUG
	int ii,jj;
#endif

	/* initialize the other arrays used for the dynProg code */
	/*********************************************************/
	for(i=0; LIKELY(i<segLength); i++){
		_mm_store_si128(loadOpt+i,zero);
		_mm_store_si128(storeOpt+i,zero);
		_mm_store_si128(rD+i,zero);
	}

	/* looping through all the columns */
	/***********************************/

	for(j=0; LIKELY(j<dbLen); j++){


		/* compute the opt and cd score depending on the previous column
		 *******************************************************************
		 * set the column deletion score to zero, has to be fixed later on */
		vCD = zero;

		/* set the opt score to the elements computed in the previous column*/
		/* set the low of storeOpt to MaxS[j]                               */
		vStoreOpt = _mm_load_si128(storeOpt+segLength-1);
		vStoreOpt = _mm_slli_si128(vStoreOpt, 1);

		/* compute the current profile, depending on the character in s2 */
		/*****************************************************************/
		current_profile = query->profile + db[j]*segLength;

		/* swap the old optimal score with the new one */
		/***********************************************/
		swap = storeOpt;
		storeOpt = loadOpt;
		loadOpt = swap;

		/* main loop computing the max, precomputing etc. */
		/**************************************************/
		for(i=0; LIKELY(i<segLength); i++){
			vRD = _mm_load_si128(rD+i);
			vRD = _mm_subs_epu8(vRD, vDelIncr);
			vTmp = _mm_load_si128(loadOpt+i);
			vTmp = _mm_subs_epu8(vTmp,vDelFixed);
			vRD = _mm_max_epu8(vRD,vTmp);
			_mm_store_si128(rD+i, vRD);

			/* add the profile the prev. opt */
			vStoreOpt = _mm_adds_epu8(vStoreOpt, *(current_profile+i));
			vStoreOpt = _mm_subs_epu8(vStoreOpt, vBias);

			/* update the maxscore found so far */
			vMaxScore = _mm_max_epu8(vMaxScore, vStoreOpt);

			/* compute the correct opt score of the cell */
			vStoreOpt = _mm_max_epu8(vStoreOpt, vRD);
			vStoreOpt = _mm_max_epu8(vStoreOpt, vCD);

			/* store the opt score of the cell */
			_mm_store_si128(storeOpt+i, vStoreOpt);

			/* precompute cd for next iteration */
			vStoreOpt = _mm_subs_epu8(vStoreOpt, vDelFixed);
			vCD = _mm_subs_epu8(vCD, vDelIncr);
			vCD = _mm_max_epu8(vCD, vStoreOpt);

			/* load precomputed opt for next iteration */
			vStoreOpt = _mm_load_si128(loadOpt+i);
		}


		for(i=0;LIKELY(i<16);++i){
			int k;
		        /* compute the gap extend penalty for the current cell */
		        vCD = _mm_slli_si128(vCD,1);

			for(k=0;LIKELY(k<segLength);++k) {
			   /* compute the current optimal value of the cell */
			   vStoreOpt = _mm_load_si128(storeOpt+k);
			   vStoreOpt = _mm_max_epu8(vStoreOpt,vCD);
			   _mm_store_si128(storeOpt+k,vStoreOpt);

			   /* precompute the scores for the next cell */
			   vStoreOpt = _mm_subs_epu8(vStoreOpt,vDelFixed);
			   vCD = _mm_subs_epu8(vCD, vDelIncr);

			   if(UNLIKELY(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_subs_epu8(vCD,vStoreOpt),zero)) == 0xFFFF)) goto shortcut;
			}
		}
                shortcut:

#ifdef DEBUG
		debug("%c\t",db[j]);
		for(ii=0; ii<16;++ii) {
		   for(jj=0; jj<segLength;++jj) {
		      if(ii*segLength+jj < query->len)
			debug("%d\t",(int)((unsigned char*)storeOpt)[ii+jj*16]);
		   }
		}
		debug("\n");
#endif

		/* store the new MaxScore for the next line block */
		/**************************************************/

		/* store the element of storeOpt in MaxS */
		vStoreOpt = _mm_load_si128(storeOpt+segLength-1);
	}

	vMaxScore = _mm_max_epu8(vMaxScore, _mm_srli_si128(vMaxScore, 8));
	vMaxScore = _mm_max_epu8(vMaxScore, _mm_srli_si128(vMaxScore, 4));
	vMaxScore = _mm_max_epu8(vMaxScore, _mm_srli_si128(vMaxScore, 2));
	vMaxScore = _mm_max_epu8(vMaxScore, _mm_srli_si128(vMaxScore, 1));
	MaxScore = (unsigned char)_mm_extract_epi16(vMaxScore,0);
	if ((int)MaxScore + (int)query->bias >=255)
		return DBL_MAX;
	return((double)MaxScore);
}


EXPORT void swps3_freeProfileByteSSE( ProfileByte * profile ){
	free( profile );
}

