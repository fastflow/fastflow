/** \file DynProgr_altivec.cc
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

#include "matrix.h"
#include "DynProgr_altivec.h"
#include <cstdlib>
#include <malloc.h>
#include <float.h>
#include <cstdio>
#include <string.h>
#include <altivec.h>
#include <sys/types.h>

#define ALIGN16(x)    (((x)+15)&(-16))
#undef SHORTCUT

template<typename T> static inline T min( T a, T b ){ return a<b?a:b; }
template<typename T> static inline T max( T a, T b ){ return a>b?a:b; }

template<typename T> struct IsInteger	{ static const int value = false; };
template<> struct IsInteger<u_int8_t>	{ static const int value = true; };
template<> struct IsInteger<u_int16_t>	{ static const int value = true; };
template<> struct IsInteger<u_int32_t>	{ static const int value = true; };
template<> struct IsInteger<int8_t>	{ static const int value = true; };
template<> struct IsInteger<int16_t>	{ static const int value = true; };
template<> struct IsInteger<int32_t>	{ static const int value = true; };

template<typename T> struct IsSigned	{ static const int value = (T)-1 < (T)0; };

template<typename T> struct MaxValue	{ static const T value = IsSigned<T>::value ? -1ll ^ (1ll<<(sizeof(T)*8-1)) : (T)-1; };
template<> struct MaxValue<float>	{ static const float value = FLT_MAX; };
template<> struct MaxValue<double>	{ static const double value = DBL_MAX; };

template<typename T> struct MinValue	{ static const T value = IsSigned<T>::value ? 1ll<<(sizeof(T)*8-1) : (T)0; };
template<> struct MinValue<float>	{ static const float value = FLT_MIN; };
template<> struct MinValue<double>	{ static const double value = DBL_MIN; };

template<typename T, typename V> struct Profile {
	int len;
	T bias;
	V * rD;
	V * storeOpt;
	V * loadOpt;
	V * profile;
};

/**
 * Allocates a profile data structure.
 * 
 * \param T The datatype for the profile.
 * \param V The datatype of the corresponding packed vectors.
 * \param len The query length.
 * \return A profile data structure.
 */
template<typename T, typename V> static inline Profile<T,V>* allocateProfile(int len)
{
	const int nSeg = sizeof(V)/sizeof(T);    // the number of segments
	const int segLen = ALIGN16(len)/nSeg; // the segment length

	Profile<T,V> *profile = (Profile<T,V>*)malloc(sizeof(*profile));
	profile->len = len;
	profile->rD = (V*)malloc(sizeof(V)*segLen);
	profile->loadOpt = (V*)malloc(sizeof(V)*segLen);
	profile->storeOpt = (V*)malloc(sizeof(V)*segLen);
	profile->profile = (V*)malloc(sizeof(V)*MATRIX_DIM*segLen);
	return profile;
}

/**
 * Releases memory occupied by a profile data structure.
 */
template<typename T, typename V> void freeProfile(Profile<T,V> *profile)
{
	free(profile->profile);
	free(profile->storeOpt);
	free(profile->loadOpt);
	free(profile->rD);
	free(profile);
}

/**
 * Saturated addition for integer data types
 */
template<typename V> static inline V vec_addx(V a, V b)
{
	return vec_adds(a,b);
}

/**
 * Saturated addition for floating point data types
 */
typedef vector float v_float_t;
template<> /*static*/ inline v_float_t vec_addx<v_float_t>(v_float_t a, v_float_t b)
{
	return vec_add(a,b);
}

/**
 * Saturated subtraction for integer data types
 */
template<typename V> static inline V vec_subx(V a, V b)
{
	return vec_subs(a,b);
}

/**
 * Saturated substraction for floating point data types
 */
typedef vector float v_float_t;
template<> /*static*/ inline v_float_t vec_subx<v_float_t>(v_float_t a, v_float_t b)
{
	return vec_sub(a,b);
}


/**
 * Performs an local alignment using the given profile and database sequence.
 *
 * \param db		The database sequence.
 * \param dbLen		The length of the database sequence.
 * \param profile	The profile created previously from the query sequence.
 * \param options	Some global options.
 * \return		The local alignment score or a maximum value if result
 * 			exceeds threshold.
 */
template< typename T, typename V > static inline T dynProgrLocal(
		const char* db, int dbLen,
		Profile<T,V> * profile,
		Options *options){

	/**********************************************************************
	* This version of the code implements the idea presented in
	*
	***********************************************************************
	* Striped Smith-Waterman speeds database searches six times over other
	* SIMD implementations
	*
	* Michael Farrar, Bioinformatics, 23(2), pp. 156-161, 2007
	**********************************************************************/

	T zero,goal;
	/* A vectorized template version */
	if (IsInteger<T>::value){
		// adjust the zero and goal values...
		zero = MinValue<T>::value + profile->bias;
		goal = MaxValue<T>::value - profile->bias;
	} else {
		zero = (T)0.0;
		goal = MaxValue<T>::value;
	}

	V vZero = {zero};
	vZero = vec_splat( vZero, 0 );
	V vGoal = {goal};
	vGoal = vec_splat( vGoal, 0 );
	V vDelFixed = {(T)options->gapOpen};
	vDelFixed = vec_splat( vDelFixed, 0 );
	V vDelInc = {(T)options->gapExt};
	vDelInc = vec_splat( vDelInc, 0 );
	V vBias = {(T)profile->bias};
	vBias = vec_splat( vBias, 0 );

	T maxScore = zero;
	const int nSeg = sizeof(V)/sizeof(T);    // the number of segments
	const int segLen = ALIGN16(profile->len)/nSeg; // the segment length

	V vMaxScore = vZero;                     // The maximum score
	/* Initialize the other arrays */
	/*******************************/
	for(int i=0; LIKELY(i<segLen); i++)
		profile->loadOpt[i] = profile->storeOpt[i] = profile->rD[i] = vZero;

	/* looping through all the columns */
	/***********************************/
	for( int i=0; LIKELY(i<dbLen); i++ ){
		V vCD = vZero;

		// set the opt score to the elements computed in the previous column
		V vStoreOpt = vec_sld(vZero, profile->storeOpt[segLen-1], sizeof(V)-sizeof(T));

		/* compute the current profile, depending on the character in s2 */
		/*****************************************************************/
		V * currentProfile = profile->profile + db[i]*segLen;

#if 0
		for(int ii=0; ii<nSeg; ++ii) {
			for(int jj=0; jj<segLen; ++jj) {
				if(ii*segLen+jj < profile->len)
					printf("\t%d",(int)((T*)currentProfile)[ii+jj*nSeg]);
			}
		}
		printf("\n");
#endif

		/* swap the old optimal score with the new one */
		/***********************************************/
		V * swap = profile->storeOpt;
		profile->storeOpt = profile->loadOpt;
		profile->loadOpt  = swap;

		/* main loop computing the max, precomputing etc. */
		/**************************************************/
		for( int j=0; LIKELY(j<segLen); j++ ){
			// Load the the rd value
			V vRD          = profile->rD[j];
			V vTmp         = profile->loadOpt[j];
			vRD            = vec_addx(vRD,vDelInc);
			vTmp           = vec_addx(vTmp,vDelFixed);
			if(!IsInteger<T>::value || !IsSigned<T>::value) {
				vRD    = vec_max(vRD,vZero);
			}
			vRD            = vec_max(vTmp,vRD);
			profile->rD[j] = vRD;
 
			// add the profile the prev. opt
			vStoreOpt = vec_addx(currentProfile[j],vStoreOpt);
			if(!IsSigned<T>::value)
				vStoreOpt = vec_subx(vStoreOpt,vBias);

			// update the maxscore found so far
			vMaxScore = vec_max( vMaxScore, vStoreOpt );
			// precompute the maximum here
			vTmp = vec_max( vCD, vRD );
			// compute the correct opt score of the cell
			vStoreOpt = vec_max( vStoreOpt, vTmp );

			// store the opt score of the cell
			profile->storeOpt[j] = vStoreOpt;

			// precompute rd and cd for next iteration
			vStoreOpt = vec_addx(vStoreOpt,vDelFixed);
			vRD       = vec_addx(vRD,vDelInc);
			vCD       = vec_addx(vCD,vDelInc);
			if(!IsInteger<T>::value || !IsSigned<T>::value)
				vStoreOpt = vec_max(vStoreOpt, vZero);
			vRD       = vec_max( vStoreOpt, vRD );
			vCD       = vec_max( vStoreOpt, vCD );

			// store precomputed rd
			profile->rD[j] = vRD;

			// load precomputed opt for next iteration
			vStoreOpt = profile->loadOpt[j];
		}

		/* TODO prefetch next profile into cache */

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

		V vStoreOptx = profile->storeOpt[0];
		vStoreOptx = vec_addx(vStoreOptx,vDelFixed - vDelInc);
		if(!IsInteger<T>::value)
			vStoreOptx = vec_max( vStoreOptx, vZero );
		V vCDx = vec_sld(vZero, vCD, sizeof(V)-sizeof(T));

		if(vec_all_le(vCDx,vStoreOptx) == 0) {
			for(int j=0; LIKELY(j<nSeg); ++j) {
				// set everything up for the next iteration
				vCD = vec_sld(vZero, vCD, sizeof(V)-sizeof(T));

				for(int k=0; LIKELY(k<segLen-1); ++k) {
					// compute the current optimal value of the cell
					vStoreOpt = profile->storeOpt[k];
					vStoreOpt = vec_max( vStoreOpt, vCD );
					profile->storeOpt[k] = vStoreOpt;

					// precompute the scores for the next cell
					vCD = vec_addx( vCD, vDelInc);
					vStoreOpt = vec_addx( vStoreOpt, vDelFixed);
					if(!IsInteger<T>::value) {
						vCD = vec_max( vCD, vZero );
						vStoreOpt = vec_max( vStoreOpt, vZero );
					}

					#ifdef SHORTCUT
					if(UNLIKELY(vec_all_le(vCD,vStoreOpt)))
						goto shortcut;
					#endif
				}

				// compute the current optimal value of the cell
				vStoreOpt = profile->storeOpt[segLen-1];
				vStoreOpt = vec_max( vStoreOpt, vCD );
				profile->storeOpt[segLen-1] = vStoreOpt;

				// precompute the cd value for the next cell
				vCD = vec_addx( vCD, vDelInc);
				vStoreOpt = vec_addx( vStoreOpt, vDelFixed);
				if(!IsInteger<T>::value) {
					vCD = vec_max( vCD, vZero );
					vStoreOpt = vec_max( vStoreOpt, vZero );
				}

				if(UNLIKELY(vec_all_le(vCD,vStoreOpt)))
					break;
			}
			#ifdef SHORTCUT
			shortcut:
			(void)1;
			#endif
		}

#ifdef DEBUG
		printf("%c\t",db[i]);
		for(int ii=0; ii<nSeg; ++ii) {
			for(int jj=0; jj<segLen; ++jj) {
				if(ii*segLen+jj < profile->len)
					printf("%d\t",(int)(((T*)profile->storeOpt)[ii+jj*nSeg]-zero));
			}
		}
		printf("\n");
#endif
	}
	return maxScore;
}

/**
 * Unrolled version of <code>dynProgrLocal</code>
 *
 * \param db		The database sequence.
 * \param dbLen		The length of the database sequence.
 * \param profile	The profile created previously from the query sequence.
 * \param options	Some global options.
 * \return		The local alignment score or a maximum value if result
 * 			exceeds threshold.
 * \see dynProgrLocal()
 * \see swps3_createProfileAltivec()
 */
template< typename T, typename V > static inline T dynProgrLocal2(
		const char* db, int dbLen,
		Profile<T,V> * profile,
		Options *options){
	/**********************************************************************
	* This version of the code implements the idea presented in
	*
	***********************************************************************
	* Striped Smith-Waterman speeds database searches six times over other
	* SIMD implementations
	*
	* Michael Farrar, Bioinformatics, 23(2), pp. 156-161, 2007
	**********************************************************************/

	T zero,goal;
	/* A vectorized template version */
	if (IsInteger<T>::value){
		// adjust the zero and goal values...
		zero = MinValue<T>::value + profile->bias;
		goal = MaxValue<T>::value - profile->bias;
	} else {
		zero = (T)0.0;
		goal = MaxValue<T>::value;
	}

	V vZero = {zero};
	vZero = vec_splat( vZero, 0 );
	V vGoal = {goal};
	vGoal = vec_splat( vGoal, 0 );
	V vDelFixed = {(T)options->gapOpen};
	vDelFixed = vec_splat( vDelFixed, 0 );
	V vDelInc = {(T)options->gapExt};
	vDelInc = vec_splat( vDelInc, 0 );
	V vBias = {(T)profile->bias};
	vBias = vec_splat( vBias, 0 );

	T maxScore = zero;
	const int nSeg = sizeof(V)/sizeof(T);    // the number of segments
	const int segLen = (ALIGN16(profile->len)/nSeg + 1) & ~1; // the segment length
	const int subSegLen = segLen / 2;                 // the sub segment length

	V vMaxScore1 = vZero, vMaxScore2 = vZero; // The maximum score

	/* Initialize the other arrays */
	/*******************************/
	for(int i=0; LIKELY(i<segLen); i++)
		profile->loadOpt[i] = profile->storeOpt[i] = profile->rD[i] = vZero;

	/* looping through all the columns */
	/***********************************/
	for( int i=0; LIKELY(i<dbLen); i++ ){
		V vCD1 = vZero, vCD2 = vZero;

		// set the opt score to the elements computed in the previous column
		V vStoreOpt1 = vec_sld(vZero, profile->storeOpt[segLen-1], sizeof(V)-sizeof(T));
		V vStoreOpt2 = profile->storeOpt[subSegLen-1];

		/* compute the current profile, depending on the character in s2 */
		/*****************************************************************/
		V * currentProfile = profile->profile + db[i]*segLen;


#if 0
		for(int ii=0; ii<nSeg; ++ii) {
			for(int jj=0; jj<segLen; ++jj) {
				if(ii*segLen+jj < profile->len)
					printf("\t%d",(int)((T*)currentProfile)[ii+jj*nSeg]);
			}
		}
		printf("\n");
#endif

		/* swap the old optimal score with the new one */
		/***********************************************/
		V * swap = profile->storeOpt;
		profile->storeOpt = profile->loadOpt;
		profile->loadOpt  = swap;

		/* main loop computing the max, precomputing etc. */
		/**************************************************/
		for( int j=0; LIKELY(j<subSegLen); j++ ){
			// Load the the rd value
			V vRD1          = profile->rD[j          ];
			V vRD2          = profile->rD[j+subSegLen];
			V vTmp1         = profile->loadOpt[j          ];
			V vTmp2         = profile->loadOpt[j+subSegLen];
			vRD1            = vec_addx(vRD1,vDelInc);
			vRD2            = vec_addx(vRD2,vDelInc);
			vTmp1           = vec_addx(vTmp1,vDelFixed);
			vTmp2           = vec_addx(vTmp2,vDelFixed);
			if(!IsInteger<T>::value || !IsSigned<T>::value) {
				vRD1    = vec_max(vRD1,vZero);
				vRD2    = vec_max(vRD2,vZero);
			}
			vRD1            = vec_max(vTmp1,vRD1);
			vRD2            = vec_max(vTmp2,vRD2);
			profile->rD[j          ] = vRD1;
			profile->rD[j+subSegLen] = vRD2;

			// add the profile the prev. opt
			vStoreOpt1 = vec_addx(currentProfile[j          ],vStoreOpt1);
			vStoreOpt2 = vec_addx(currentProfile[j+subSegLen],vStoreOpt2);
			if(!IsSigned<T>::value) {
				vStoreOpt1 = vec_subx(vStoreOpt1,vBias);
				vStoreOpt2 = vec_subx(vStoreOpt2,vBias);
			}

			// update the maxscore found so far
			vMaxScore1 = vec_max( vMaxScore1, vStoreOpt1 );
			vMaxScore2 = vec_max( vMaxScore2, vStoreOpt2 );

			// precompute the maximum here
			vTmp1 = vec_max( vCD1, vRD1 );
			vTmp2 = vec_max( vCD2, vRD2 );

			// compute the correct opt score of the cell
			vStoreOpt1 = vec_max( vStoreOpt1, vTmp1 );
			vStoreOpt2 = vec_max( vStoreOpt2, vTmp2 );

			// store the opt score of the cell
			profile->storeOpt[j          ] = vStoreOpt1;
			profile->storeOpt[j+subSegLen] = vStoreOpt2;

			// precompute rd and cd for next iteration
			vStoreOpt1 = vec_addx(vStoreOpt1,vDelFixed);
			vStoreOpt2 = vec_addx(vStoreOpt2,vDelFixed);
			vRD1       = vec_addx(vRD1,vDelInc);
			vRD2       = vec_addx(vRD2,vDelInc);
			vCD1       = vec_addx(vCD1,vDelInc);
			vCD2       = vec_addx(vCD2,vDelInc);
			if(!IsInteger<T>::value || !IsSigned<T>::value) {
				vStoreOpt1 = vec_max(vStoreOpt1, vZero);
				vStoreOpt2 = vec_max(vStoreOpt2, vZero);
			}
			vRD1       = vec_max( vStoreOpt1, vRD1 );
			vRD2       = vec_max( vStoreOpt2, vRD2 );
			vCD1       = vec_max( vStoreOpt1, vCD1 );
			vCD2       = vec_max( vStoreOpt2, vCD2 );

			// store precomputed rd
			profile->rD[j          ] = vRD1;
			profile->rD[j+subSegLen] = vRD2;

			// load precomputed opt for next iteration
			vStoreOpt1 = profile->loadOpt[j          ];
			vStoreOpt2 = profile->loadOpt[j+subSegLen];
		}

		/* TODO prefetch next profile into cache */

		/* set totcells */
		/****************/
//         totcells += ls1;
		/* check for a changed MaxScore */
		/********************************/
		V vMaxScore = vec_max( vMaxScore1, vMaxScore2 );
		for( T* tmp = (T*)&vMaxScore; tmp<(T*)(&vMaxScore+1); tmp++ )
			if (UNLIKELY(maxScore < *tmp))
				maxScore = *tmp;
		// if the goal was reached, exit
		if ( UNLIKELY(maxScore >= goal) )
			return MaxValue<T>::value;

		V vStoreOptx1 = profile->storeOpt[0        ];
		V vStoreOptx2 = profile->storeOpt[subSegLen];
		vStoreOptx1 = vec_addx(vStoreOptx1,vDelFixed - vDelInc);
		vStoreOptx2 = vec_addx(vStoreOptx2,vDelFixed - vDelInc);
		if(!IsInteger<T>::value) {
			vStoreOptx1 = vec_max( vStoreOptx1, vZero );
			vStoreOptx2 = vec_max( vStoreOptx2, vZero );
		}
		V vCDx1 = vec_sld(vZero, vCD2, sizeof(V)-sizeof(T));
		V vCDx2 = vCD1;

		if(UNLIKELY((vec_all_le(vCDx1,vStoreOptx1) == 0) || (vec_all_le(vCDx2,vStoreOptx2) == 0))) {
			for(int j=0; LIKELY(j<nSeg+1); ++j) {
				// set everything up for the next iteration
				V vRotate = vCD2;
				vCD2 = vCD1;
				vCD1 = vec_sld(vZero, vRotate, sizeof(V)-sizeof(T));

				for(int k=0; LIKELY(k<subSegLen-1); ++k) {
					// compute the current optimal value of the cell
					vStoreOpt1 = profile->storeOpt[k            ];
					vStoreOpt2 = profile->storeOpt[k + subSegLen];
					vStoreOpt1 = vec_max( vStoreOpt1, vCD1 );
					vStoreOpt2 = vec_max( vStoreOpt2, vCD2 );
					profile->storeOpt[k            ] = vStoreOpt1;
					profile->storeOpt[k + subSegLen] = vStoreOpt2;

					// precompute the scores for the next cell
					vCD1 = vec_addx( vCD1, vDelInc);
					vCD2 = vec_addx( vCD2, vDelInc);
					vStoreOpt1 = vec_addx( vStoreOpt1, vDelFixed);
					vStoreOpt2 = vec_addx( vStoreOpt2, vDelFixed);
					if(!IsInteger<T>::value) {
						vCD1 = vec_max( vCD1, vZero );
						vCD2 = vec_max( vCD2, vZero );
						vStoreOpt1 = vec_max( vStoreOpt1, vZero );
						vStoreOpt2 = vec_max( vStoreOpt2, vZero );
					}

					#ifdef SHORTCUT
					if(UNLIKELY(vec_all_le(vCD1,vStoreOpt1) != 0 && vec_all_le(vCD2,vStoreOpt2) != 0))
						goto shortcut;
					#endif
				}

				// compute the current optimal value of the cell
				vStoreOpt1 = profile->storeOpt[subSegLen - 1];
				vStoreOpt2 = profile->storeOpt[segLen    - 1];
				vStoreOpt1 = vec_max( vStoreOpt1, vCD1 );
				vStoreOpt2 = vec_max( vStoreOpt2, vCD2 );
				profile->storeOpt[subSegLen - 1] = vStoreOpt1;
				profile->storeOpt[segLen    - 1] = vStoreOpt2;

				// precompute the scores for the next cell
				vCD1 = vec_addx( vCD1, vDelInc);
				vCD2 = vec_addx( vCD2, vDelInc);
				vStoreOpt1 = vec_addx( vStoreOpt1, vDelFixed);
				vStoreOpt2 = vec_addx( vStoreOpt2, vDelFixed);
				if(!IsInteger<T>::value) {
					vCD1 = vec_max( vCD1, vZero );
					vCD2 = vec_max( vCD2, vZero );
					vStoreOpt1 = vec_max( vStoreOpt1, vZero );
					vStoreOpt2 = vec_max( vStoreOpt2, vZero );
				}

				if(UNLIKELY(vec_all_le(vCD1,vStoreOpt1) != 0 && vec_all_le(vCD2,vStoreOpt2) != 0))
					break;
			}
			#ifdef SHORTCUT
			shortcut:
			(void)1;
			#endif
		}
#ifdef DEBUG
		printf("%c\t",db[i]);
		for(int ii=0; ii<nSeg; ++ii) {
			for(int jj=0; jj<segLen; ++jj) {
				if(ii*segLen+jj < profile->len)
					printf("%d\t",(int)(((T*)profile->storeOpt)[ii+jj*nSeg]-zero));
			}
		}
		printf("\n");
#endif
	}
	return maxScore;
}

/**
 * Template version for allocation and computation of a profile for the given
 * data type and similarity matrix.
 *
 * \param T		The data type for the alignment scores.
 * \param V		The corresponding packed vector data type.
 * \param X		The data type of the similarity matrix.
 * \param query		The query sequence.
 * \param queryLen	The length of the query sequence.
 * \param simi		The similarity matrix.
 *
 * \return		An initialized profile data structure.
 */
template< typename T, typename V, typename X >
EXPORT Profile<T,V>* swps3_createProfileAltivec( const char *query, int queryLen, X* simi ){
	const int alignedLen = ALIGN16(queryLen);
	const int nSeg = sizeof(V)/sizeof(T);    // the number of segments
	const int segLen = alignedLen/nSeg; // the segment length

	Profile<T,V>* profile = allocateProfile<T,V>(queryLen);

	for( int i=0; i<MATRIX_DIM; i++ ){
		T *currentProfile = ((T*)profile->profile)+i*alignedLen;
		for( int j=0; j<segLen; j++ ){
			T *tmp = currentProfile + j*nSeg;
			for( int k=0; k<nSeg; k++ )
				if( j + k*segLen < queryLen )
					tmp[k] = (T)simi[ query[j + k*segLen ] * MATRIX_DIM + i ];
				else
					tmp[k] = 0;
		}
	}

	return profile;
}

/**
 * Template version of alignment routine for Altivec. To be called from C++
 * code.
 *
 * \param T		The data type for the alignment scores.
 * \param V		The corresponding packed vector data type.
 * \param db		The database sequence.
 * \param dbLen		The length of the database sequence.
 * \param profile	A profile data structure previously computed from the
 * 			query sequence.
 * \param options	Some global options.
 *
 * \return		The local alignment score or <code>DBL_MAX</code> if
 * 			score exceeds threshold.
 *
 * \see swps3_createProfileAltivec()
 */
template< typename T, typename V >
EXPORT double swps3_dynProgrAltivec(const char *db, int dbLen, Profile<T,V> *profile, Options *options){
	T zero, goal;
	/* A vectorized template version */
	if (IsInteger<T>::value){
		// adjust the zero and goal values...
		zero = MinValue<T>::value;
		goal = MaxValue<T>::value;
	} else {
		zero = (T)0.0;
		goal = MaxValue<T>::value;
	}

	T maxScore=zero;

#ifdef UNROLL
	T currentScore;
	if (sizeof(T) < 2)
		currentScore = dynProgrLocal<T,V> ( db, dbLen, profile, options );
	else
		currentScore = dynProgrLocal2<T,V> ( db, dbLen, profile, options );
#else
	T currentScore = dynProgrLocal<T,V> ( db, dbLen, profile, options );
#endif
	if( maxScore < currentScore)
		maxScore = currentScore;

	if(maxScore >= goal)
		return DBL_MAX;

	/* Finally free all the memory we allocated */
	/********************************************/
	return (double)(maxScore-zero);
}

/**
 * C version of alignment routine for Altivec for signed 8-bit integers.
 *
 * \param db		The database sequence.
 * \param dbLen		The length of the database sequence.
 * \param profile	A profile data structure previously computed from the
 * 			query sequence.
 * \param options	Some global options.
 *
 * \return		The local alignment score or <code>DBL_MAX</code> if
 * 			score exceeds threshold.
 *
 * \see swps3_createProfileByteAltivec()
 * \see swps3_dynProgrAltivec()
 */
EXPORT double swps3_dynProgrByteAltivec(const char *db, int dbLen, void* profile, Options *options)
{
	return swps3_dynProgrAltivec<int8_t,vector int8_t>(db,dbLen,(Profile<int8_t,vector int8_t>*)profile,options);
}

/**
 * C version of alignment routine for Altivec for signed 16-bit integers.
 *
 * \param db		The database sequence.
 * \param dbLen		The length of the database sequence.
 * \param profile	A profile data structure previously computed from the
 * 			query sequence.
 * \param options	Some global options.
 *
 * \return		The local alignment score or <code>DBL_MAX</code> if
 * 			score exceeds threshold.
 *
 * \see swps3_createProfileShortAltivec()
 * \see swps3_dynProgrAltivec()
 */
EXPORT double swps3_dynProgrShortAltivec(const char *db, int dbLen, void* profile, Options *options)
{
	return swps3_dynProgrAltivec<int16_t,vector int16_t>(db,dbLen,(Profile<int16_t,vector int16_t>*)profile,options);
}

/**
 * C version of alignment routine for Altivec for signed 32-bit floating point
 * values.
 *
 * \param db		The database sequence.
 * \param dbLen		The length of the database sequence.
 * \param profile	A profile data structure previously computed from the
 * 			query sequence.
 * \param options	Some global options.
 *
 * \return		The local alignment score or <code>DBL_MAX</code> if
 * 			score exceeds threshold.
 *
 * \see swps3_createProfileFloatAltivec()
 * \see swps3_dynProgrAltivec()
 */
EXPORT double swps3_dynProgrFloatAltivec(const char *db, int dbLen, void* profile, Options *options)
{
	return swps3_dynProgrAltivec<float,vector float>(db,dbLen,(Profile<float,vector float>*)profile,options);
}

/**
 * C version for allocation and computation of a 8-bit integer profile for the
 * given data type and similarity matrix.
 *
 * \param query		The query sequence.
 * \param queryLen	The length of the query sequence.
 * \param matrix	The similarity matrix.
 *
 * \return		An initialized profile data structure.
 */
EXPORT void *swps3_createProfileByteAltivec(const char *query, int queryLen, SBMatrix matrix)
{
	return swps3_createProfileAltivec<int8_t, vector int8_t>(query, queryLen, matrix);
}

/**
 * C version for allocation and computation of a 16-bit integer profile for the
 * given data type and similarity matrix.
 *
 * \param query		The query sequence.
 * \param queryLen	The length of the query sequence.
 * \param matrix	The similarity matrix.
 *
 * \return		An initialized profile data structure.
 */
EXPORT void *swps3_createProfileShortAltivec(const char *query, int queryLen, SBMatrix matrix)
{
	return swps3_createProfileAltivec<int16_t, vector int16_t>(query, queryLen, matrix);
}

/**
 * C version for allocation and computation of a 32-bit floating point profile
 * for the given data type and similarity matrix.
 *
 * \param query		The query sequence.
 * \param queryLen	The length of the query sequence.
 * \param matrix	The similarity matrix.
 *
 * \return		An initialized profile data structure.
 */
EXPORT void *swps3_createProfileFloatAltivec(const char *query, int queryLen, SBMatrix matrix)
{
	return swps3_createProfileAltivec<float, vector float>(query, queryLen, matrix);
}

/**
 * C version for deallocation of a 8-bit integer profile.
 *
 * \param profile	A profile data structure.
 */
EXPORT void swps3_freeProfileByteAltivec(void *profile)
{
  freeProfile<int8_t, vector int8_t>((Profile<int8_t, vector int8_t> *)profile);
}

/**
 * C version for deallocation of a 16-bit integer profile.
 *
 * \param profile	A profile data structure.
 */
EXPORT void swps3_freeProfileShortAltivec(void *profile)
{
  freeProfile<int16_t, vector int16_t>((Profile<int16_t, vector int16_t> *)profile);
}

/**
 * C version for deallocation of a 32-bit floating point profile.
 *
 * \param profile	A profile data structure.
 */
EXPORT void swps3_freeProfileFloatAltivec(void *profile)
{
  freeProfile<float, vector float>((Profile<float, vector float> *)profile);
}
