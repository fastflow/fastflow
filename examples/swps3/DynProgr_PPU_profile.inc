/*
 * Copyright (c) 2007-2008 Philipp Kraehenbuehl, Adam Szalkowski
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

int blockStart;
int nSeg = 16/sizeof(T); /* The number of elements in the vector */
SPEProfile *profile = malloc(sizeof(*profile));
T * pp1 = (T*)memalign( 16, MATRIX_DIM*(queryLen+16)*sizeof(T));

profile->len = queryLen;
profile->addr = (ppu_addr_t)pp1;
profile->min = DBL_MAX;
profile->max = DBL_MIN;

/* Calculate the Block Size */
/****************************/

/* The block size needs to be aligned to the number of elements in the vector (nSeg) */
if (TOTAL_MEMORY<maxDbLen*(sizeof(char)+2*sizeof(T))) error("insufficient memory on SPE");
profile->blockSize = (TOTAL_MEMORY-maxDbLen*(sizeof(char)+2*sizeof(T)))/((MATRIX_DIM+3)*sizeof(T)) & -16;
if (profile->blockSize<=50) error("insufficient memory on SPE");
profile->blockSize = ALIGN16( MIN(profile->blockSize,queryLen) );
/* Setup the profile */
/*********************/
for(blockStart=0; blockStart<queryLen; blockStart+=profile->blockSize){
	T * currentBlock = pp1+blockStart*MATRIX_DIM;
	/* The current block size also needs to be aligned to nSeg */
	int currentBlockSize = ALIGN16( MIN(profile->blockSize,queryLen-blockStart) );
	int segLen = currentBlockSize / nSeg;
	int i,j,k;

	/* Calculate the alignment for the current block */
	for( i=0; i<MATRIX_DIM; i++ ){
		T * currentProfile = currentBlock+i*currentBlockSize;
		for( j=0; nSeg*j < currentBlockSize; j++ ){
			T * tmp = currentProfile + j*nSeg;
			for( k=0; k<nSeg; k++ )
				if( j + k*segLen + blockStart < queryLen ) {
					tmp[k] = (T)matrix[ query[j + k*segLen + blockStart] * MATRIX_DIM + i ];
					if(profile->min > tmp[k]) profile->min = tmp[k];
					if(profile->max < tmp[k]) profile->max = tmp[k];
				}
				else tmp[k] = 0;
		}
	}
}

return profile;
