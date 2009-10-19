/** \file DynProgr_SPE_functions.cc
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

#include "DynProgr_SPE_functions.h"
#include "DynProgr_SPE.h"
#include "matrix.h"
#include <cstdlib>
#include <malloc.h>
#include <float.h>
#include <cstdio>
#include <string.h>
#include <spu_intrinsics.h>
#include <sys/types.h>

template<typename T> static inline T min( T a, T b ){ return a<b?a:b; }
template<typename T> static inline T max( T a, T b ){ return a>b?a:b; }

template<typename T> struct IsInteger	{ static const bool value = false; };
template<> struct IsInteger<int8_t>	{ static const bool value = true; };
template<> struct IsInteger<int16_t>	{ static const bool value = true; };
template<> struct IsInteger<int32_t>	{ static const bool value = true; };

template<typename T> struct MaxValue	{ static const T value = -1 ^ (1ll<<(sizeof(T)*8-1)); };
template<> struct MaxValue<float>	{ static const float value = FLT_MAX; };
template<> struct MaxValue<double>	{ static const double value = DBL_MAX; };

template<typename T> struct MinValue	{ static const T value = 1ll<<(sizeof(T)*8-1); };
template<> struct MinValue<float>	{ static const float value = FLT_MIN; };
template<> struct MinValue<double>	{ static const double value = DBL_MIN; };

char * s1, * s2;
int ls1, ls2;
int blockStart, blockSize;
int maxDbLen;
double mn, mx, fixedDel, incDel;
void *simi, *profile, *loadOpt, *storeOpt, *rD, *maxS, *delS;
ppu_addr_t remote_profile;

/**
 * Emulates a packed maximum operation.
 */
template< class V > static inline V spu_max( V a, V b ){
	return spu_sel(a,b,spu_cmpgt(b,a));
}
/**
 * Emulates a packed minimum operation.
 */
template< class V > static inline V spu_min( V a, V b ){
	return spu_sel(a,b,spu_cmpgt(a,b));
}

#undef SHORTCUT

/**
 * Performs an local alignment using the given profile segment and database sequence.
 *
 * \param currentBlockSize 	Length of the current profile segment in terms
 * 				of query sequence characters.
 * \param zero			Defines a value for the zero score.
 * \param goal			Defines a maximum value for the score.
 * \param maxS			Intermediate row with maximum scores.
 * \param delS			Intermediate row with column deletion scores.
 * \param profile		A segment of the profile created previously from the query sequence.
 * \param loadOpt		Temporary storage for a column.
 * \param storeOpt		Temporary storage for a column.
 * \param rD			Temporary storage for a column of row deletion
 * 				scores.
 * \return			The local alignment score or a maximum value if result
 * 				exceeds threshold.
 */
template< class T, class V > static inline T dynProgrLocalBlock(
										int currentBlockSize,
										T zero, T goal,
										T * maxS, T* delS,
										const V * profile,
										V * loadOpt,
										V * storeOpt,
										V * rD){
	/**********************************************************************
	* This version of the code implements the idea presented in
	*
	***********************************************************************
	* Striped Smith-Waterman speeds database searches six times over other
	* SIMD implementations
	*
	* Michael Farrar, Bioinformatics, 23(2), pp. 156-161, 2007
	**********************************************************************/

	const V vZero = spu_splats( zero );
	const V vGoal = spu_splats( goal );
	const V vDelFixed = spu_splats( (T)fixedDel );
	const V vDelInc   = spu_splats( (T)incDel   );

	T maxScore = zero;
	const int nSeg = sizeof(V)/sizeof(T);    // the number of segments
	const int segLen = currentBlockSize/nSeg; // the segment length

	V vMaxScore = vZero;                     // The maximum score
	T prevMax   = zero;
	/* Initialize the other arrays */
	/*******************************/
	for(int i=0; LIKELY(i<segLen); i++)
		loadOpt[i] = storeOpt[i] = rD[i] = vZero;

	/* looping through all the columns */
	/***********************************/
	for( int i=0; LIKELY(i<ls2); i++ ){

		/* compute the opt and cd score depending on the previous column */
		/*******************************************************************/
		// set the column deletion score to zero, has to be fixed later on
		V vCD = spu_insert( delS[i], vZero, 0);

		// set the opt score to the elements computed in the previous column
		// set the low of storeOpt to MaxS[j]
		// spu_rlmaskqwbyte is a complicated way to say right shift
		V vStoreOpt = spu_rlmaskqwbyte(storeOpt[segLen-1], -sizeof(T));
		vStoreOpt = spu_insert( prevMax, vStoreOpt, 0 );

		/* compute the current profile, depending on the character in s2 */
		/*****************************************************************/
		const V * currentProfile = profile + s2[i]*segLen;

#if 0
		for(int ii=0; ii<nSeg; ++ii) {
			for(int jj=0; jj<segLen; ++jj) {
				if(ii*segLen+jj < ls1)
					printf("\t%d",(int)((T*)currentProfile)[ii+jj*nSeg]);
			}
		}
		printf("\n");
#endif

		/* swap the old optimal score with the new one */
		/***********************************************/
		V * swap = storeOpt;
		storeOpt = loadOpt;
		loadOpt  = swap;

		/* main loop computing the max, precomputing etc. */
		/**************************************************/
		for( int j=0; LIKELY(j<segLen); j++ ){
			// Load the the rd value
			V vRD           = rD[j];
			V vTmp          = loadOpt[j];
			vRD            += vDelInc;
			vTmp           += vDelFixed;
			if(IsInteger<T>::value) {
				vRD     = spu_max(vRD,vZero);
			}
			vRD             = spu_max(vTmp,vRD);
			rD[j]           = vRD;

			// add the profile the prev. opt
			vStoreOpt += currentProfile[j];

			// To provide saturated arithmetics
			if (IsInteger<T>::value)
				vStoreOpt = spu_min( vStoreOpt, vGoal );

			// update the maxscore found so far
			vMaxScore = spu_max( vMaxScore, vStoreOpt );
			// precompute the maximum here
			vTmp = spu_max( vCD, vRD );
			// compute the correct opt score of the cell
			vStoreOpt = spu_max( vStoreOpt, vTmp );
			vStoreOpt = spu_max( vStoreOpt, vZero );

			// store the opt score of the cell
			storeOpt[j] = vStoreOpt;

			// precompute rd and cd for next iteration
			vStoreOpt += vDelFixed;
			vCD       += vDelInc;
			if(IsInteger<T>::value)
				vStoreOpt = spu_max( vStoreOpt, vZero );
			vCD       = spu_max( vStoreOpt, vCD );

			// load precomputed opt for next iteration
			vStoreOpt = loadOpt[j];
		}

		/* set totcells */
		/****************/
//         totcells += ls1;
		/* check for a changed MaxScore */
		/********************************/
		for( T* tmp = (T*)&vMaxScore; tmp<(T*)(&vMaxScore+1); tmp++ )
			if (UNLIKELY(maxScore < *tmp))
				maxScore = *tmp;
		// if the goal was reached, exit
		if ( UNLIKELY(maxScore >= goal) )
			return MaxValue<T>::value;

		/* cleaning up with missed arrows */
		/**********************************/
		delS[i] = spu_extract( vCD, nSeg-1 );

		V vStoreOptx = storeOpt[0];
		vStoreOptx = spu_max(vStoreOptx + (vDelFixed - vDelInc),vZero);
		V vCDx = spu_rlmaskqwbyte(vCD, -sizeof(T));
		vCDx = spu_insert( zero, vCDx, 0 );

		if( spu_extract(spu_gather((vector unsigned char)spu_cmpgt(vCDx,vStoreOptx)),0) != 0) {
			for(int j=0; LIKELY(j<nSeg); ++j) {
				// set everything up for the next iteration
				vCD = spu_rlmaskqwbyte(vCD, -sizeof(T));
				vCD = spu_insert( zero, vCD, 0 );

				for(int k=0; k<segLen-1; ++k) {
					// compute the current optimal value of the cell
					vStoreOpt = storeOpt[k];
					vStoreOpt = spu_max( vStoreOpt, vCD );
					storeOpt[k] = vStoreOpt;

					// precompute the scores for the next cell
					vCD = spu_max( vCD + vDelInc, vZero );
					vStoreOpt = spu_max( vStoreOpt + vDelFixed, vZero );

					#ifdef SHORTCUT
					if(UNLIKELY(spu_extract(spu_gather((vector unsigned char)spu_cmpgt(vCD,vStoreOpt)),0) == 0))
						goto shortcut;
					#endif

				}

				// compute the current optimal value of the cell
				vStoreOpt = storeOpt[segLen-1];
				vStoreOpt = spu_max( vStoreOpt, vCD );
				storeOpt[segLen-1] = vStoreOpt;

				// precompute the cd value for the next cell
				vCD = spu_max( vCD + vDelInc, vZero );
				vStoreOpt = spu_max( vStoreOpt + vDelFixed, vZero );

				// Update the del Score
				T temp = spu_extract( vCD, nSeg-1 );
				if ( UNLIKELY(delS[i] < temp) )
					delS[i] = temp;

				if(UNLIKELY(spu_extract(spu_gather((vector unsigned char)spu_cmpgt(vCD,vStoreOpt)),0) == 0)) break;
			}
			#ifdef SHORTCUT
			shortcut:
			(void)1;
			#endif
		}

		/* store the new MaxScore for the next line block */
		/**************************************************/
		prevMax = maxS[i];
		maxS[i] = spu_extract( storeOpt[segLen-1], nSeg-1 );

#ifdef DEBUG
		printf("%c\t",s2[i]);
		for(int ii=0; ii<nSeg; ++ii) {
			for(int jj=0; jj<segLen; ++jj) {
				if(ii*segLen+jj < ls1)
					printf("%d\t",(int)(((T*)storeOpt)[ii+jj*nSeg]-zero));
			}
		}
		printf("\n");
#endif
	}
	return maxScore;
}

#ifdef UNROLL
/**
 * Performs an local alignment using the given profile segment and database
 * sequence. This version of the code has been unrolled once for performance
 * reasons.
 *
 * \param currentBlockSize 	Length of the current profile segment in terms
 * 				of query sequence characters.
 * \param zero			Defines a value for the zero score.
 * \param goal			Defines a maximum value for the score.
 * \param maxS			Intermediate row with maximum scores.
 * \param delS			Intermediate row with column deletion scores.
 * \param profile		A segment of the profile created previously from the query sequence.
 * \param loadOpt		Temporary storage for a column.
 * \param storeOpt		Temporary storage for a column.
 * \param rD			Temporary storage for a column of row deletion
 * 				scores.
 * \return			The local alignment score or a maximum value if result
 * 				exceeds threshold.
 */
template< class T, class V > static inline T dynProgrLocalBlock2(
										int currentBlockSize,
										T zero, T goal,
										T * maxS, T* delS,
										const V * profile,
										V * loadOpt,
										V * storeOpt,
										V * rD){
	/**********************************************************************
	* This version of the code implements the idea presented in
	*
	***********************************************************************
	* Striped Smith-Waterman speeds database searches six times over other
	* SIMD implementations
	*
	* Michael Farrar, Bioinformatics, 23(2), pp. 156-161, 2007
	**********************************************************************/

	const V vZero = spu_splats( zero );
	const V vGoal = spu_splats( goal );
	const V vDelFixed = spu_splats( (T)fixedDel );
	const V vDelInc   = spu_splats( (T)incDel   );

	T maxScore = zero;
	const int nSeg = sizeof(V)/sizeof(T);             // the number of segments
	const int segLen = (currentBlockSize/nSeg + 1) & ~1; // the segment length
	const int subSegLen = segLen / 2;                 // the sub segment length
	V vMaxScore1 = vZero,vMaxScore2 = vZero;  // The maximum score
	T prevMax   = zero;

	/* Initialize the other arrays */
	/*******************************/
	for(int i=0; LIKELY(i<segLen); i++)
		loadOpt[i] = storeOpt[i] = rD[i] = vZero;

	/* looping through all the columns */
	/***********************************/
	for( int i=0; LIKELY(i<ls2); i++ ){

		/* compute the opt and cd score depending on the previous column */
		/*******************************************************************/
		// set the column deletion score to zero, has to be fixed later on
		V vCD1 = spu_insert( delS[i], vZero, 0);
		V vCD2 = vZero;

		// set the opt score to the elements computed in the previous column
		// set the low of storeOpt to MaxS[j]
		// spu_rlmaskqwbyte is a complicated way to say right shift
		V vStoreOpt1 = spu_rlmaskqwbyte(storeOpt[segLen-1], -sizeof(T));
		vStoreOpt1 = spu_insert( prevMax, vStoreOpt1, 0 );
		V vStoreOpt2 = storeOpt[subSegLen-1];
		/* compute the current profile, depending on the character in s2 */
		/*****************************************************************/
		const V * currentProfile = profile + s2[i]*segLen;

#if 0
		for(int ii=0; ii<nSeg; ++ii) {
			for(int jj=0; jj<segLen; ++jj) {
				if(ii*segLen+jj < ls1)
					printf("\t%d",(int)((T*)currentProfile)[ii+jj*nSeg]);
			}
		}
		printf("\n");
#endif

		/* swap the old optimal score with the new one */
		/***********************************************/
		V * swap     = storeOpt;
		storeOpt = loadOpt;
		loadOpt  = swap;

		/* main loop computing the max, precomputing etc. */
		/**************************************************/
		for( int j=0; LIKELY(j<subSegLen); j++ ){
			// lead the row deletion score
			V vRD1           = rD[j];
			V vRD2           = rD[j+subSegLen];
			V vTmp1          = loadOpt[j];
			V vTmp2          = loadOpt[j+subSegLen];
			vRD1            += vDelInc;
			vRD2            += vDelInc;
			vTmp1           += vDelFixed;
			vTmp2           += vDelFixed;
			if(IsInteger<T>::value) {
				vRD1     = spu_max(vRD1,vZero);
				vRD2     = spu_max(vRD2,vZero);
			}
			vRD1             = spu_max(vTmp1,vRD1);
			vRD2             = spu_max(vTmp2,vRD2);
			rD[j]            = vRD1;
			rD[j+subSegLen]  = vRD2;

			// add the profile the prev. opt
			vStoreOpt1 += currentProfile[j];
			vStoreOpt2 += currentProfile[j+subSegLen];

			// To avoid saturated arithmetics
			if (IsInteger<T>::value){
				vStoreOpt1 = spu_min( vStoreOpt1, vGoal );
				vStoreOpt2 = spu_min( vStoreOpt2, vGoal );
			}
			// update the maxscore found so far
			vMaxScore1 = spu_max( vMaxScore1, vStoreOpt1 );
			vMaxScore2 = spu_max( vMaxScore2, vStoreOpt2 );

			// precompute the maximum here (gives about 5% speedup)
			vTmp1 = spu_max( vCD1, vRD1 );
			vTmp2 = spu_max( vCD2, vRD2 );
			// compute the correct opt score of the cell
			vStoreOpt1 = spu_max( vStoreOpt1, vTmp1 );
			vStoreOpt2 = spu_max( vStoreOpt2, vTmp2 );
			vStoreOpt1 = spu_max( vStoreOpt1, vZero );
			vStoreOpt2 = spu_max( vStoreOpt2, vZero );

			// store the opt score of the cell
			storeOpt[j          ] = vStoreOpt1;
			storeOpt[j+subSegLen] = vStoreOpt2;

			// precompute rd and cd for next iteration
			vStoreOpt1 += vDelFixed;
			vStoreOpt2 += vDelFixed;
			vCD1       += vDelInc;
			vCD2       += vDelInc;
			if(IsInteger<T>::value) {
				vStoreOpt1 = spu_max( vStoreOpt1, vZero );
				vStoreOpt2 = spu_max( vStoreOpt2, vZero );
			}
			vCD1 = spu_max( vStoreOpt1, vCD1 );
			vCD2 = spu_max( vStoreOpt2, vCD2 );

			// load precomputed opt for next iteration
			vStoreOpt1 = loadOpt[j];
			vStoreOpt2 = loadOpt[j+subSegLen];
		}

		/* set totcells */
		/****************/
//         totcells += ls1;
		/* check for a changed MaxScore */
		/********************************/
		V vMaxScore = spu_max( vMaxScore1, vMaxScore2 );
		for( T* tmp = (T*)&vMaxScore; tmp<(T*)(&vMaxScore+1); tmp++ )
			if (UNLIKELY(maxScore < *tmp))
				maxScore = *tmp;
		// if the goal was reached, exit
		if ( UNLIKELY(maxScore >= goal) )
			return MaxValue<T>::value;

		/* cleaning up with missed arrows */
		/**********************************/
		delS[i] = spu_extract( vCD2, nSeg-1 );

		V vStoreOptx1 = storeOpt[0        ];
		V vStoreOptx2 = storeOpt[subSegLen];
		vStoreOptx1 = spu_max(vStoreOpt1 + (vDelFixed - vDelInc),vZero);
		vStoreOptx2 = spu_max(vStoreOpt2 + (vDelFixed - vDelInc),vZero);
		V vCDx2 = vCD1;
		V vCDx1 = spu_rlmaskqwbyte(vCD2, -sizeof(T));
		vCDx1 = spu_insert( zero, vCDx1, 0 );
		if( spu_extract(spu_gather(spu_or((vector unsigned char)spu_cmpgt(vCDx1,vStoreOptx1),(vector unsigned char)spu_cmpgt(vCDx2,vStoreOptx2))),0) != 0) {
			for(int j=0; LIKELY(j<nSeg+1); ++j) {
				// set everything up for the next iteration
				V vRotate = vCD2;
				vCD2 = vCD1;
				vCD1 = spu_rlmaskqwbyte(vRotate, -sizeof(T));
				vCD1 = spu_insert( zero, vCD1, 0 );

				for(int k=0; k<subSegLen-1; ++k) {
					// compute the current optimal value of the cell
					vStoreOpt1 = storeOpt[k            ];
					vStoreOpt2 = storeOpt[k + subSegLen];
					vStoreOpt1 = spu_max( vStoreOpt1, vCD1 );
					vStoreOpt2 = spu_max( vStoreOpt2, vCD2 );
					storeOpt[k            ] = vStoreOpt1;
					storeOpt[k + subSegLen] = vStoreOpt2;

					// precompute the scores for the next cell
					vStoreOpt1 = spu_max( vStoreOpt1 + vDelFixed, vZero );
					vStoreOpt2 = spu_max( vStoreOpt2 + vDelFixed, vZero );
					vCD1 = spu_max( vCD1 + vDelInc, vZero );
					vCD2 = spu_max( vCD2 + vDelInc, vZero );

					#ifdef SHORTCUT
					if(UNLIKELY(spu_extract(spu_gather(spu_or((vector unsigned char)spu_cmpgt(vCD1,vStoreOpt1),(vector unsigned char)spu_cmpgt(vCD2,vStoreOpt2))),0) == 0))
						goto shortcut;
					#endif

				}

				// compute the current optimal value of the cell
				vStoreOpt1 = storeOpt[subSegLen - 1];
				vStoreOpt2 = storeOpt[segLen    - 1];
				vStoreOpt1 = spu_max( vStoreOpt1, vCD1 );
				vStoreOpt2 = spu_max( vStoreOpt2, vCD2 );
				storeOpt[subSegLen - 1] = vStoreOpt1;
				storeOpt[segLen    - 1] = vStoreOpt2;

				// precompute the scores for the next cell
				vStoreOpt1 = spu_max( vStoreOpt1 + vDelFixed, vZero );
				vStoreOpt2 = spu_max( vStoreOpt2 + vDelFixed, vZero );
				vCD1 = spu_max( vCD1 + vDelInc, vZero );
				vCD2 = spu_max( vCD2 + vDelInc, vZero );

				// Update the del Score
				T temp = spu_extract( vCD2, nSeg-1 );
				if ( UNLIKELY(delS[i] < temp) )
					delS[i] = temp;

				if(UNLIKELY(spu_extract(spu_gather(spu_or((vector unsigned char)spu_cmpgt(vCD1,vStoreOpt1),(vector unsigned char)spu_cmpgt(vCD2,vStoreOpt2))),0) == 0)) break;
			}
			#ifdef SHORTCUT
			shortcut:
			(void)1;
			#endif
		}

		/* store the new MaxScore for the next line block */
		/**************************************************/
		prevMax = maxS[i];
		maxS[i] = spu_extract( storeOpt[segLen-1], nSeg-1 );

#ifdef DEBUG
		printf("%c\t",s2[i]);
		for(int ii=0; ii<nSeg; ++ii) {
			for(int jj=0; jj<segLen; ++jj) {
				if(ii*segLen+jj < ls1)
					printf("%d\t",(int)(((T*)storeOpt)[ii+jj*nSeg]-zero));
			}
		}
		printf("\n");
#endif
	}
	return maxScore;
}
#endif

/**
 * Computes a profile and writes it to the specified memory location.
 *
 * \param blockStart		Current position in the query sequence.
 * \param currentBlockSize 	Length of current profile segment.
 * \param simi			The similarity matrix.
 * \param currentBlock		Memory location to write the profile to.
 */
template< class T, class V >
static void doCreateProfile( int blockStart, int currentBlockSize, const T* simi, V* currentBlock){
	const int nSeg = sizeof(V)/sizeof(T);    // the number of segments
	const int segLen = currentBlockSize/nSeg; // the segment length

	for( int i=0; i<MATRIX_DIM; i++ ){
		T *currentProfile = ((T*)currentBlock)+i*currentBlockSize;
		for( int j=0; j<segLen; j++ ){
			T *tmp = currentProfile + j*nSeg;
			for( int k=0; k<nSeg; k++ )
				if( j + k*segLen + blockStart < ls1 )
					tmp[k] = simi[ s1[j + k*segLen + blockStart] * MATRIX_DIM + i ];
				else
					tmp[k] = 0;
		}
	}
}

/**
 * Template for creating a profile using global variables as parameters.
 *
 * \param T		The data type for the alignment scores.
 * \param V		The corresponding packed vector data type.
 */
template< class T, class V >
static void TcreateProfile(void){
	doCreateProfile( blockStart, ALIGN16(min(ls1-blockStart, blockSize)), (T*)simi, (V*)profile);
}

/**
 * Template for computing a local alignment score using global variables as
 * parameters.
 *
 * \param T		The data type for the alignment scores.
 * \param V		The corresponding packed vector data type.
 */
template< class T, class V >
static double TdynProgLocal(void){
	T zero, goal;
	/* A vectorized template version */
	if (IsInteger<T>::value){
		// adjust the zero and goal values...
		zero = MinValue<T>::value;
		zero-= mn;
		goal = MaxValue<T>::value;
		goal-= mx;
	} else {
		zero = 0.0;
		goal = MaxValue<T>::value;
	}

	/* Set the stored max and del score to zero */
	/********************************************/
	for( int i=0; i<ls2; i++ )
		((T*)maxS)[i] = ((T*)delS)[i] = (T)zero;

	T maxScore=zero;
	blockStart = 0;
	do {
		const int currentBlockSize = ALIGN16(min(ls1-blockStart,blockSize));
		/* initialize the profile for the current iteration */
		if(blockStart != 0) { /* when blockStart==0 then the profile has been initialized already */
			if(remote_profile == 0) {
#ifdef DEBUG_FETCH
				printf(">>>> creating profile\n");
#endif
				doCreateProfile<T,V>( blockStart, currentBlockSize, (T*)simi, (V*)profile);
			} else {
#ifdef DEBUG_FETCH
				printf(">>>> fetching profile (%lu bytes)\n",  currentBlockSize * MATRIX_DIM * sizeof(T));
#endif
				for( int64_t bs=0; bs<currentBlockSize * MATRIX_DIM * sizeof(T); bs+=MAX_TRANSFER ) {
					mfc_get( ((char*)profile)+bs, remote_profile+blockStart*MATRIX_DIM*sizeof(T)+bs, ALIGN16(min(currentBlockSize*MATRIX_DIM*sizeof(T)-bs, (int64_t)MAX_TRANSFER)), 0, 0, 0 );

					/* wait for DMA to finish */
					mfc_write_tag_mask(1<<0);
					mfc_read_tag_status_all();
				}
			}
		}

#ifdef UNROLL
		T currentScore;
		if (sizeof(T) < 2)
			currentScore = dynProgrLocalBlock<T,V> ( currentBlockSize, zero, goal, (T*)maxS, (T*)delS, (V*)profile, (V*)loadOpt, (V*)storeOpt, (V*)rD );
		else
			currentScore = dynProgrLocalBlock2<T,V> ( currentBlockSize, zero, goal, (T*)maxS, (T*)delS, (V*)profile, (V*)loadOpt, (V*)storeOpt, (V*)rD );
#else
		T currentScore = dynProgrLocalBlock<T,V> ( currentBlockSize, zero, goal, (T*)maxS, (T*)delS, (V*)profile, (V*)loadOpt, (V*)storeOpt, (V*)rD );
#endif
		if( maxScore < currentScore)
			maxScore = currentScore;

		if(maxScore >= goal)
			return DBL_MAX;

		if(blockStart+blockSize >= ls1) break;

		blockStart += blockSize;
	} while(1);
	/* Finally free all the memory we allocated */
	/********************************************/
	return (double)(maxScore-zero);
}

/**
 * A structure for accessing different flavors of the local alignment routine.
 */
dvf_t dynProgLocal[] = {
	TdynProgLocal<int8_t, vector int8_t>,
	TdynProgLocal<int16_t, vector int16_t>,
	TdynProgLocal<int32_t, vector int32_t>,
	TdynProgLocal<float, vector float>,
	TdynProgLocal<double, vector double>
};

/**
 * A structure for accessing different flavors of the profile creation routine.
 */
vvf_t createProfile[] = {
	TcreateProfile<int8_t, vector int8_t>,
	TcreateProfile<int16_t, vector int16_t>,
	TcreateProfile<int32_t, vector int32_t>,
	TcreateProfile<float, vector float>,
	TcreateProfile<double, vector double>
};

