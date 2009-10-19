/** \file DynProgr_SPE.h
 *
 * Definitions for message handling and memory management on the SPE.
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

#ifndef DYNPROGR_SPE_H_
#define DYNPROGR_SPE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>

#define TOTAL_MEMORY 200000
#define MAX_TRANSFER 16384

#define __ALIGNED__ __attribute__((__aligned__(16)))
#define ALIGN16(x)    (((x)+15)&(-16))
#define ALIGN32(x)    (((x)+31)&(-32))
/* This structs need to use long long instead of type* as *
 * pointer, because SPE is 32 bit and PPE 64              */

/*All types should be less than 16 */
#define SPE_ALIGNMENT_TYPE_MASK 0xf
#define SPE_ALIGNMENT_TYPE_DOUBLE 5
#define SPE_ALIGNMENT_TYPE_FLOAT  4
#define SPE_ALIGNMENT_TYPE_INT    3
#define SPE_ALIGNMENT_TYPE_SHORT  2
#define SPE_ALIGNMENT_TYPE_BYTE   1

/* 16 is the special flag for the precalculated profile version */
#define SPE_ALIGNMENT_PROFILE  16

typedef unsigned long long ppu_addr_t;

typedef struct{
    int32_t len;
    int32_t blockSize;
    double min;
    double max;
    ppu_addr_t addr;
} SPEProfile;

typedef struct{
    int32_t len;
    ppu_addr_t addr;
} SPESequence;

typedef struct{
    double min;
    double max;
    ppu_addr_t addr;
} SPEMatrix;

enum SPECommands{
	SPE_CMD_INIT,
	SPE_CMD_CREATE_PROFILE,
	SPE_CMD_PUT_PROFILE,
	SPE_CMD_GET_PROFILE,
	SPE_CMD_ALIGN
};

enum SPEDatatypes{
	SPE_DATA_INT8 = 0,
	SPE_DATA_INT16 = 1,
	SPE_DATA_INT32 = 2,
	SPE_DATA_FLOAT = 3,
	SPE_DATA_DOUBLE = 4
};

static const int dataSize[5] = {1,2,4,sizeof(float),sizeof(double)};

typedef struct{
	int32_t command;
	union {
		struct {
			int32_t datatype;
			int32_t dbMaxLen;
			double fixedDel;
			double incDel;
		} INIT;
		struct {
			SPESequence query;
			SPEMatrix matrix;
		} CREATE_PROFILE;
		struct {
			ppu_addr_t addr;
			int32_t blockSize; /*out*/
		} PUT_PROFILE;
		struct {
			SPEProfile profile;
		} GET_PROFILE;
		struct {
			SPESequence db;
			double result; /*out*/
		} ALIGN;
	} data;
} SPECommand;

/* Memory allocation functions */
void * alloc( int n );
void reset();
int memRemaining();

#ifdef __cplusplus
}
#endif

#endif
