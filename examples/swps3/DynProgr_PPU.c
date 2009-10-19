/** \file DynProgr_PPU.c
 *
 * PPU frontend to the alignment routines on the SPU.
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

#include <stdio.h>
#include <malloc.h>
#ifdef MAIL
#include <libspe.h>
#include <sched.h>
#else
#include <libspe2.h>
#endif
#include <sys/wait.h>
#include <sys/types.h>
#include <errno.h>
#include "DynProgr_SPE.h"
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "debug.h"
#include "swps3.h"
#include "matrix.h"
#include "DynProgr_scalar.h"
#include "DynProgr_PPU.h"

#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)

extern spe_program_handle_t spe_dynprogr_handle;

#ifdef MAIL
static spe_gid_t getSpeGid(){
	static spe_gid_t gid = 0;
	if (gid == 0){
		gid = spe_create_group(SCHED_RR, 1, 1);
		if (gid <= 0)
			error("PPU:Creating new group failed");
	}
	return gid;
}
/** Acquires a SPE. */
static speid_t getSpeId(){
	static speid_t id = 0;
	if (!id)
		id = spe_create_thread(getSpeGid(), &spe_dynprogr_handle, NULL, NULL, -1, 0);
	return id;
}

/** sends a 64-bit value to the SPE mailbox. */
static void writeMailBox( ppu_addr_t addr ){
	speid_t spe_id = getSpeId();
	spe_write_in_mbox( spe_id, (uint32_t)addr );
	spe_write_in_mbox( spe_id, (uint32_t)(addr>>32) );
}

/** reads a 32-bit value from the SPE mailbox. */
static int readMailBox(){
	struct spe_event e =  {
		.events = SPE_EVENT_MAILBOX,
		.gid = getSpeGid(),
    };
	if( !spe_get_event(&e, 1, -1) || e.revents !=  SPE_EVENT_MAILBOX ) error("PPU:Waiting for MailBox signal");
	return e.data;
}
#else
static spe_context_ptr_t spe_context = NULL;
#endif

/** Sends a SPECommand to the SPE.
 *
 * \param command A SPEcommand.
 * \return Returns nonzero on error.
 */
static int submitSPECommand(SPECommand* command) {
#ifdef MAIL
	/* Call the SPU Program*/
	writeMailBox( (ppu_addr_t)command );
	return readMailBox();
#else
	if( !spe_context ){
		spe_context = spe_context_create( 0, NULL );
		spe_program_load( spe_context, &spe_dynprogr_handle );
	}
	
	unsigned int entry = SPE_DEFAULT_ENTRY;
	return spe_context_run( spe_context, &entry, 0, command, NULL, NULL );
	/* spe_context_destroy( spe_context ); */
#endif
}

/** Initializes a SPE.
 *
 * \param dataType The data type for the computation. Must be one of
 * SPE_ALIGNMENT_TYPE_DOUBLE, SPE_ALIGNMENT_TYPE_FLOAT, SPE_ALIGNMENT_TYPE_INT,
 * SPE_ALIGNMENT_TYPE_SHORT, SPE_ALIGNMENT_TYPE_BYTE.
 * \param maxDbLen Maximal possible length of the database sequence.
 * \param options Program options.
 *
 * \return Nonzero on error.
 */
static int SPEInit(int dataType, int maxDbLen, Options *options) {
	SPECommand command __ALIGNED__;

	command.command = SPE_CMD_INIT;
	command.data.INIT.dbMaxLen = maxDbLen;
	command.data.INIT.datatype = dataType;
	command.data.INIT.fixedDel = options->gapOpen;
	command.data.INIT.incDel = options->gapExt;

	return submitSPECommand(&command);
}

/** Creates a profile in the SPE's memory.
 *
 * \param query The query sequence.
 * \param queryLen The length of the query sequence.
 * \param matrix The scoring matrix with entries of the same type as specified
 * in the SPEInit command.
 * \param min The highest value in the scoring matrix.
 * \param max The lowest value in the scoring matrix.
 *
 * \see SPEInit()
 *
 * \return Nonzero on error.
 */
static int SPECreateProfile(const char* query, int queryLen, const void* matrix, double min, double max) {
	SPECommand command __ALIGNED__;
	assert(((ppu_addr_t)query & 0xf) == 0 && "query not aligned to 16 bytes");
	assert(((ppu_addr_t)matrix & 0xf) == 0 && "matrix not aligned to 16 bytes");

	command.command = SPE_CMD_CREATE_PROFILE;
	command.data.CREATE_PROFILE.query.addr = (ppu_addr_t)query;
	command.data.CREATE_PROFILE.query.len = queryLen;
	command.data.CREATE_PROFILE.matrix.addr = (ppu_addr_t)matrix;
	command.data.CREATE_PROFILE.matrix.min = min;
	command.data.CREATE_PROFILE.matrix.max = max;

	return submitSPECommand(&command);
}

static int SPEGetProfile(const SPEProfile *profile) {
	SPECommand command __ALIGNED__;
	command.command = SPE_CMD_GET_PROFILE;

	assert(((ppu_addr_t)profile->addr & 0xf) == 0 && "profile not aligned to 16 bytes");
	memcpy(&command.data.GET_PROFILE.profile, profile, sizeof(*profile));

	return submitSPECommand(&command);
}

static int SPEPutProfile(void *profile, int *blockSize) {
	int ret;
	SPECommand command __ALIGNED__;
	assert(((ppu_addr_t)profile & 0xf) == 0 && "profile not aligned to 16 byte");

	command.command = SPE_CMD_PUT_PROFILE;
	command.data.PUT_PROFILE.blockSize = 0;
	command.data.PUT_PROFILE.addr = (ppu_addr_t)profile;

	ret =  submitSPECommand(&command);

	/* write back result */
	*blockSize = command.data.PUT_PROFILE.blockSize;

	return ret;
}

static int SPEAlign(const char *db, int dbLen, double *result) {
	int ret;
	SPECommand command __ALIGNED__;
	assert(((ppu_addr_t)db & 0xf) == 0 && "db sequence not aligned to 16 byte");

	command.command = SPE_CMD_ALIGN;
	command.data.ALIGN.result = -1;
	command.data.ALIGN.db.len = dbLen;
	command.data.ALIGN.db.addr = (ppu_addr_t)db;

	ret =  submitSPECommand(&command);

	/* write back result */
	*result = command.data.ALIGN.result;

	return ret;
}

EXPORT double swps3_alignmentByteSPE( const SBMatrix matrix, const char * query, int queryLen, const char * db, int dbLen, Options * options ) {
	int i, ret;
	double min, max, maxScore;

	ret = SPEInit(SPE_DATA_INT8, dbLen, options);
	if(ret != 0) error("error in SPEInit");

	/* Setup the DayMatrix */
	max = MAX(options->gapExt,options->gapOpen);
	min = MIN(options->gapExt,options->gapOpen);
	for(i=0; i<MATRIX_DIM*MATRIX_DIM; i++) {
		if(max < matrix[i]) max = matrix[i];
		if(min > matrix[i]) min = matrix[i];
	}
	ret =  SPECreateProfile(query, queryLen, matrix, min, max);
	if(ret != 0) error("error in SPECreateProfile");

	ret = SPEAlign(db, dbLen, &maxScore);
	if(ret != 0) error("error in SPEAlign");

	/* If something failed use the ppu version */
	if( ret != 0 || maxScore == -1 ){
		double dmatrix[MATRIX_DIM*MATRIX_DIM];
		warning( "sequence too long for SPE program, using PPU fallback\n" );
		for(i=0; i<MATRIX_DIM*MATRIX_DIM; ++i) dmatrix[i] = matrix[i];
		maxScore = swps3_alignScalar( dmatrix, query, queryLen, db, dbLen, options );
	}
	return maxScore;
}

EXPORT double swps3_alignmentShortSPE( const SBMatrix matrix, const char * query, int queryLen, const char * db, int dbLen, Options * options ) {
	int i, ret;
	double max, min, maxScore;
	int16_t simi[MATRIX_DIM*MATRIX_DIM] __ALIGNED__;

	ret = SPEInit(SPE_DATA_INT16, dbLen, options);
	if(ret != 0) error("error in SPEInit");
	
	/* Setup the DayMatrix */
	max = MAX(options->gapExt,options->gapOpen);
	min = MIN(options->gapExt,options->gapOpen);
	for(i=0; i<MATRIX_DIM*MATRIX_DIM; i++) {
		if(max < matrix[i]) max = matrix[i];
		if(min > matrix[i]) min = matrix[i];
		simi[i] = matrix[i];
	}
	ret =  SPECreateProfile(query, queryLen, simi, min, max);
	if(ret != 0) error("error in SPECreateProfile");

	ret = SPEAlign(db, dbLen, &maxScore);
	if(ret != 0) error("error in SPEAlign");

	/* If something failed use the ppu version */
	if( ret != 0 || maxScore == -1 ){
		double dmatrix[MATRIX_DIM*MATRIX_DIM];
		warning( "sequence too long for SPE program, using PPU fallback\n" );
		for(i=0; i<MATRIX_DIM*MATRIX_DIM; ++i) dmatrix[i] = matrix[i];
		maxScore = swps3_alignScalar( dmatrix, query, queryLen, db, dbLen, options );
	}
	return maxScore;
}

EXPORT double swps3_alignmentFloatSPE( const SBMatrix matrix, const char * query, int queryLen, const char * db, int dbLen, Options * options ) {
	int i, ret;
	double max, min, maxScore;
	float simi[MATRIX_DIM*MATRIX_DIM] __ALIGNED__;

	ret = SPEInit(SPE_DATA_FLOAT, dbLen, options);
	if(ret != 0) error("error in SPEInit");
	
	/* Setup the DayMatrix */
	max = MAX(options->gapExt,options->gapOpen);
	min = MIN(options->gapExt,options->gapOpen);
	for(i=0; i<MATRIX_DIM*MATRIX_DIM; i++) {
		if(max < matrix[i]) max = matrix[i];
		if(min > matrix[i]) min = matrix[i];
		simi[i] = (float)matrix[i];
	}
	ret =  SPECreateProfile(query, queryLen, simi, min, max);
	if(ret != 0) error("error in SPECreateProfile");

	ret = SPEAlign(db, dbLen, &maxScore);
	if(ret != 0) error("error in SPEAlign");

	/* If something failed use the ppu version */
	if( ret != 0 || maxScore == -1 ){
		double dmatrix[MATRIX_DIM*MATRIX_DIM];
		warning( "sequence too long for SPE program, using PPU fallback\n" );
		for(i=0; i<MATRIX_DIM*MATRIX_DIM; ++i) dmatrix[i] = matrix[i];
		maxScore = swps3_alignScalar( dmatrix, query, queryLen, db, dbLen, options );
	}
	return maxScore;
}

/**
 * Profile must have been loaded before!
 */
EXPORT double swps3_alignmentProfileSPE( const char * db, int dbLen )
{
	double maxScore;
	int ret;

	ret = SPEAlign(db, dbLen, &maxScore);
	if(ret != 0) maxScore = -1;

	return maxScore;
}

EXPORT void swps3_loadProfileByte(SPEProfile *profile, int maxDbLen, Options *options)
{
	int ret;

	ret = SPEInit(SPE_DATA_INT8, maxDbLen, options);
	if(ret != 0) error("error in SPEInit");

	ret = SPEGetProfile(profile);
	if(ret != 0) error("error in SPEGetProfile");
}

EXPORT void swps3_loadProfileShort(SPEProfile *profile, int maxDbLen, Options *options)
{
	int ret;

	ret = SPEInit(SPE_DATA_INT16, maxDbLen, options);
	if(ret != 0) error("error in SPEInit");

	ret = SPEGetProfile(profile);
	if(ret != 0) error("error in SPEGetProfile");
}

EXPORT void swps3_loadProfileFloat(SPEProfile *profile, int maxDbLen, Options *options)
{
	int ret;

	ret = SPEInit(SPE_DATA_FLOAT, maxDbLen, options);
	if(ret != 0) error("error in SPEInit");

	ret = SPEGetProfile(profile);
	if(ret != 0) error("error in SPEGetProfile");
}

#define T int8_t
EXPORT SPEProfile * swps3_createProfileBytePPU( const char * query, int queryLen, const SBMatrix matrix, int maxDbLen )
{
#include "DynProgr_PPU_profile.inc"
}
#undef T

#define T int16_t
EXPORT SPEProfile * swps3_createProfileShortPPU( const char * query, int queryLen, const SBMatrix matrix, int maxDbLen )
{
#include "DynProgr_PPU_profile.inc"
}
#undef T

#define T float
EXPORT SPEProfile * swps3_createProfileFloatPPU( const char * query, int queryLen, const SBMatrix matrix, int maxDbLen )
{
#include "DynProgr_PPU_profile.inc"
}
#undef T

EXPORT void swps3_freeProfilePPU(SPEProfile *profile) {
	free((void*)profile->addr);
	free(profile);
}

