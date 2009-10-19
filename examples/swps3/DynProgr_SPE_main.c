/** \file DynProgr_SPE_main.c
 *
 * Main routine and message handling on the SPE.
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

#include <spu_mfcio.h>
#include "DynProgr_SPE.h"
#include "DynProgr_SPE_functions.h"
#include <stdio.h>
#include "matrix.h"


#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

static int8_t mainMemory[ TOTAL_MEMORY ] __attribute__((__aligned__(16)));
static int allocated = 0;
static int datatype = -1;
void * alloc( int size ){
	void * result = mainMemory + allocated;
	allocated += ALIGN16(size);
	if(allocated>TOTAL_MEMORY) return (void*)-1;
	return result;
}
int memRemaining(){
	return TOTAL_MEMORY-allocated;
}
void reset(){
	allocated = 0;
	datatype = -1;
}

int handleCommand( ppu_addr_t program_data_ea ){
	SPECommand cmd __ALIGNED__;
	int i;
	/* Load the type */
	mfc_get(&cmd, program_data_ea, sizeof(cmd), 0, 0, 0);
	mfc_write_tag_mask(1<<0);
	mfc_read_tag_status_all();

	switch(cmd.command) {
	case SPE_CMD_INIT: /* resets stored data */
		reset();
		datatype = cmd.data.INIT.datatype;
		if(datatype < 0 || datatype > 4) {
			datatype = -1;
			return -1;
		}
		fixedDel = cmd.data.INIT.fixedDel;
		incDel = cmd.data.INIT.incDel;
		maxDbLen = cmd.data.INIT.dbMaxLen;

		/* reset some variables */
		profile = NULL;
		remote_profile = 0;
		blockStart = 0;
		blockSize = 0;
		s1 = NULL;
		ls1 = 0;
		simi = NULL;

		/* allocate memory for database string and inter-block
		 * buffers */
		s2 = (char *)alloc( maxDbLen*sizeof(char) );
		maxS = alloc( maxDbLen*dataSize[datatype] );
		delS = alloc( maxDbLen*dataSize[datatype] );
		break;

	case SPE_CMD_CREATE_PROFILE: /* downloads query sequence and scoring matrix and initializes the profile */
		if(profile != NULL || datatype == -1) return -1;

		mn = min(cmd.data.CREATE_PROFILE.matrix.min,min(fixedDel,incDel));
		mx = max(cmd.data.CREATE_PROFILE.matrix.max,max(fixedDel,incDel));
		ls1 = cmd.data.CREATE_PROFILE.query.len;

		/* allocate and load query sequence */
		s1 = alloc( ls1*sizeof(char) );
		for( i=0; i<ls1; i+=MAX_TRANSFER )
			mfc_get( s1+i, cmd.data.CREATE_PROFILE.query.addr+i, ALIGN16(min(ls1-i, MAX_TRANSFER)*sizeof(char)), 0, 0, 0 );

		/* allocate and load matrix */
		simi = alloc( MATRIX_DIM*MATRIX_DIM*dataSize[datatype] );
		mfc_get( simi, cmd.data.CREATE_PROFILE.matrix.addr, ALIGN16(MATRIX_DIM*MATRIX_DIM*dataSize[datatype]), 1, 0, 0 );

		/* wait for DMA to finish */
		mfc_write_tag_mask((1<<0)|(1<<1));
		mfc_read_tag_status_all();

		/* compute block size and allocate memory */
		if(memRemaining() <= 0) return -1;
		blockSize=(memRemaining() / ((MATRIX_DIM+3)*dataSize[datatype])) & -16;
		if (blockSize < 50) return -1;
		blockSize = ALIGN16(min(blockSize,ls1));

		/* allocate memory and initialize profile */
		profile  = alloc( blockSize * MATRIX_DIM * dataSize[datatype] );
		loadOpt  = alloc( blockSize * dataSize[datatype] );
		storeOpt = alloc( blockSize * dataSize[datatype] );
		rD       = alloc( blockSize * dataSize[datatype] );
		
		blockStart = 0;
#ifdef DEBUG_FETCH
		printf(">>>> creating profile\n");
#endif
		createProfile[datatype]();
		break;

	case SPE_CMD_PUT_PROFILE: /* upload profile to main memory */
		if(profile == NULL || s1 == NULL) return -1;

		/* normally we would expect the first block of the profile is
		 * already present in memory. If not generate it */
		if(blockStart != 0) {
			blockStart = 0;
			createProfile[datatype]();
		}
		cmd.data.PUT_PROFILE.blockSize = blockSize;

		/* create profile blockwise and upload it to main memory */
		for(blockStart=0; blockStart<ls1; blockStart+=blockSize ) {
			int64_t bs;
			int currentBlockSize = ALIGN16(min(ls1-blockStart,blockSize));
			if(blockStart != 0) createProfile[datatype]();

			for( bs=0; bs<currentBlockSize * MATRIX_DIM * dataSize[datatype]; bs+=MAX_TRANSFER ) {
				mfc_put( ((char*)profile)+bs, cmd.data.PUT_PROFILE.addr+blockStart*MATRIX_DIM*dataSize[datatype]+bs, ALIGN16(min(currentBlockSize*MATRIX_DIM*dataSize[datatype]-bs, (int64_t)MAX_TRANSFER)), 0, 0, 0 );

				/* wait for DMA to finish */
				mfc_write_tag_mask(1<<0);
				mfc_read_tag_status_all();
			}
		}

		/* Write back the data */
		mfc_put(&cmd, program_data_ea, sizeof(cmd), 0, 0, 0);
		mfc_write_tag_mask(1<<0);
		mfc_read_tag_status_all();
		break;

	case SPE_CMD_GET_PROFILE: /* download profile from main memory */
		if(datatype == -1 || profile != NULL) return -1;
		remote_profile = cmd.data.GET_PROFILE.profile.addr;
		
		mn = min(cmd.data.GET_PROFILE.profile.min,min(fixedDel,incDel));
		mx = max(cmd.data.GET_PROFILE.profile.max,max(fixedDel,incDel));
		ls1 = cmd.data.GET_PROFILE.profile.len;
		blockSize = cmd.data.GET_PROFILE.profile.blockSize;

		profile  = alloc( blockSize * MATRIX_DIM * dataSize[datatype] );
		loadOpt  = alloc( blockSize * dataSize[datatype] );
		storeOpt = alloc( blockSize * dataSize[datatype] );
		rD       = alloc( blockSize * dataSize[datatype] );
		if(memRemaining() < 0) return -1;

		blockStart = 0;
#ifdef DEBUG_FETCH
		printf(">>>> fetching profile (%d bytes)\n",ALIGN16(blockSize * MATRIX_DIM * dataSize[datatype]));
#endif
		for( i=0; i<ALIGN16(blockSize * MATRIX_DIM * dataSize[datatype]); i+=MAX_TRANSFER ) {
			mfc_get( ((char*)profile)+i, remote_profile+i, ALIGN16(min(blockSize*MATRIX_DIM*dataSize[datatype]-i, (int64_t)MAX_TRANSFER)), 0, 0, 0 );

			/* wait for DMA to finish */
			mfc_write_tag_mask(1<<0);
			mfc_read_tag_status_all();
		}
		break;

	case SPE_CMD_ALIGN: /* perform a local alignment */
		if(profile == NULL) return -1;

		ls2 = cmd.data.ALIGN.db.len;

		/* download database sequence */
		for( i=0; i<ls2; i+=MAX_TRANSFER )
			mfc_get( s2+i, cmd.data.ALIGN.db.addr+i, ALIGN16(min(ls2-i, MAX_TRANSFER)*sizeof(char)), 0, 0, 0 );
		mfc_write_tag_mask(1<<0);
		mfc_read_tag_status_all();

		/* initialize the profile if it has not been initialized yet */
		if(blockStart != 0) {
			if(remote_profile == 0) {
				blockStart = 0;
#ifdef DEBUG_FETCH
				printf(">>>> creating profile\n");
#endif
				createProfile[datatype]();
			} else {
				blockStart = 0;
#ifdef DEBUG_FETCH
				printf(">>>> fetching profile (%d bytes)\n",ALIGN16(blockSize * MATRIX_DIM * dataSize[datatype]));
#endif
				for( i=0; i<ALIGN16(blockSize * MATRIX_DIM * dataSize[datatype]); i+=MAX_TRANSFER ) {
					mfc_get( ((char*)profile)+i, remote_profile+i, ALIGN16(min(blockSize*MATRIX_DIM*dataSize[datatype]-i, (int64_t)MAX_TRANSFER)), 0, 0, 0 );

					/* wait for DMA to finish */
					mfc_write_tag_mask(1<<0);
					mfc_read_tag_status_all();
				}
			}
		}

		cmd.data.ALIGN.result = dynProgLocal[datatype]();

		/* Write back the data */
		mfc_put(&cmd, program_data_ea, sizeof(cmd), 0, 0, 0);
		mfc_write_tag_mask(1<<0);
		mfc_read_tag_status_all();
		break;

	default:
		return -1;
	}
	return 0;
}
#ifdef MAIL
int main() {
	while (1){
		int res;
		ppu_addr_t program_data_ea = spu_read_in_mbox();
		program_data_ea += ((ppu_addr_t)spu_read_in_mbox())<<32;
		res = handleCommand( program_data_ea );
// 		spu_write_out_mbox( res );
		spu_write_out_intr_mbox( res );
	}
	return 0;
}

#else
int main(uint64_t spe_id, ppu_addr_t program_data_ea, ppu_addr_t env) {
	(void)spe_id;
	(void)env;
	return handleCommand( program_data_ea );
}
#endif
