/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/** file swps3_ff.c
 *
 *
 * Main procedure and multi-threading code. This is the FastFlow version
 * of the swps3 code.
 *
 * Author: Massimo Torquati <torquati@di.unipi.it> <massimotor@gmail.com>
 *
 *
 */
/*
 * Copyright (c) 2007-2008 ETH Zürich, Institute of Computational Science
 *
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
#include "fasta.h"
#include "DynProgr_scalar.h"
#if defined(__SSE2__)
#include "DynProgr_sse_byte.h"
#include "DynProgr_sse_short.h"
#elif defined(__ALTIVEC__)
#include "DynProgr_altivec.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <float.h>

#include <ff/farm.hpp>
#include <ff/spin-lock.hpp>

#include <vector>
#include <fastflow.h>

using namespace ff; // FastFlow namespace

enum {EMITTER_CORE=0, EMITTER_PRIORITY=-10};
enum {INPUT_CHANNEL_SLOTS=512};

typedef struct {
    SWType  type;
    Options options;
    SBMatrix matrix;    
} args_t;


/* ------------------------- globals --------------------------------- */
lock_t lock;
unsigned int qCount,dCount,qResidues,dResidues;

/* ------------------------------------------------------------------- */



// generic worker
class Worker: public ff_node {
public:
    Worker(SWType  type, Options options, SBMatrix matrix):
        oldquery(NULL),
#ifdef __SSE2__
        profileByte(NULL), profileShort(NULL),
#elif defined(__ALTIVEC__)
        profileByte(NULL), profileShort(NULL), profileFloat(NULL),
#endif
        type(type),options(options),matrix(matrix) {}


    // called just one time at the very beginning
    int svc_init() {
        ldCount=0; ldResidues=0;

        if (ALLOCATOR && ALLOCATOR_REGISTER4FREE) {
            error("Worker, register4free fails\n");
            return -1;
        }
        
        return 0;
    }

    void * svc(void * t) {
        task_t * task = (task_t *)t;
        char * query = task->query;
        int    queryLen= task->querylen;
        double score = 0.0;

        if (task->dbLen == 0) {
            if (!query) return NULL; // this is probably a bug!

            // the query sequence is changing

#if defined(__SSE2__)
            swps3_freeProfileByteSSE(profileByte);
            swps3_freeProfileShortSSE(profileShort);
#elif defined(__ALTIVEC__)
            swps3_freeProfileByteAltivec(profileByte);
            swps3_freeProfileShortAltivec(profileShort);
            swps3_freeProfileFloatAltivec(profileFloat);
#endif
        }

        if (query != oldquery) {
            // the query has changed

#if defined(__SSE2__)
            profileByte = swps3_createProfileByteSSE( query, queryLen, matrix );
            profileShort = swps3_createProfileShortSSE( query, queryLen, matrix );	
#elif defined(__ALTIVEC__)
            profileByte = swps3_createProfileByteAltivec(query, queryLen, matrix);
            profileShort = swps3_createProfileShortAltivec(query, queryLen, matrix);
            profileFloat = swps3_createProfileFloatAltivec(query, queryLen, matrix);
#endif
            oldquery = query; 
        }

#if defined(__SSE2__)
        if(type == SSE2) {
            if( (score = swps3_alignmentByteSSE( profileByte, task->dbdata, task->dbLen, &options )) >= DBL_MAX ) {
                score = swps3_alignmentShortSSE( profileShort, task->dbdata, task->dbLen, &options );
                assert(score >= 250 && "score too low");
            }	   
        }

#elif defined(__ALTIVEC__)
        if(type == ALTIVEC) {
#if 0
            score = swps3_dynProgrFloatAltivec(task->dbdata, task->dbLen, profileFloat, &options);
#else
            score = swps3_dynProgrByteAltivec(task->dbdata, task->dbLen, profileByte, &options);
            if(score >= DBL_MAX)
                score = swps3_dynProgrShortAltivec(task->dbdata, task->dbLen, profileShort, &options);
        }
#endif
#endif /* __ALTIVEC__ */

        if(type == SCALAR)
            score = swps3_alignScalar( dmatrix, query, queryLen, task->dbdata, task->dbLen, &options);
    

        /* NOTE: Could we use write(2) and setvbuf to set the output buffer
         * in order to avoid using the following locks ?
         *
         */
        spin_lock(lock);
        if(score >= options.threshold) 
            printf(">threshold\t%s\n",task->db);
        else
            printf("%g\t%s\n",score,task->db);
        spin_unlock(lock);

        ldCount++; ldResidues+=task->dbLen;

        FREE((char*)task);

        // we don't use a collector thread so we have any task to send out
        return GO_ON; 
    }

    void  svc_end()  { 
        // updating global counters
        spin_lock(lock);
        dCount    += ldCount;
        dResidues += ldResidues;
        spin_unlock(lock);
    }

private:
    char         * oldquery;
#ifdef __SSE2__    
    ProfileByte  * profileByte;
    ProfileShort * profileShort;
#elif defined(__ALTIVEC__)
    void *profileByte, *profileShort, *profileFloat;
#endif
    double         dmatrix[MATRIX_DIM*MATRIX_DIM];
    SWType         type;
    Options        options;
    SBMatrix       matrix;  
    unsigned int   ldCount;
    unsigned int   ldResidues;
};



// the load-balancer filter
class Emitter: public ff_node {
public:
    Emitter(char * queryFile, char * dbFile):
        queryFile(queryFile),dbFile(dbFile),
        tofree(NULL),oldquery(NULL) {};
    
    // called just one time at the very beginning
    int svc_init() {
        if (ALLOCATOR && ALLOCATOR_REGISTER) {
            error("Emitter, registerAllocator fails\n");
            return -1;
        }
        lqCount=0; lqResidues=0;
        queryLib = ff_openLib( queryFile, ALLOCATOR );

        // allocate only one chunk to improve spatial locality
        task_t * task = (task_t*)MALLOC(MIN_ALLOC);
        task->db = ((char *)task)+sizeof(task_t);

        tofree= ff_readNextSequence( queryLib, task); 
        if (tofree) {
            query=task->dbdata;   queryLen=task->dbLen;

            dbLib = ff_openLib( dbFile, ALLOCATOR );
            lqCount++; lqResidues+=queryLen;            
            return 0;
        }
        return -1;
    }
    
    void * svc(void *) {
        if (!tofree) return NULL; // EOS
        
        // allocate only one chunk to improve spatial locality
        task_t * task = (task_t*)MALLOC(MIN_ALLOC);
        task->db = ((char *)task)+sizeof(task_t);

        if (!ff_readNextSequence( dbLib, task) ) {
            ff_closeLib(dbLib);

            if (oldquery) FREE(oldquery);
            oldquery=tofree;
            
            //FIX
            tofree= ff_readNextSequence( queryLib, task);

            if (tofree) {
                query=task->query;   queryLen=task->querylen;
                dbLib = ff_openLib( dbFile, ALLOCATOR );
                lqCount++; 
                return GO_ON;
            }
            FREE((char*)task);
            if (oldquery) FREE(oldquery);
            return NULL;  // EOS
        }
        task->query    = query;
        task->querylen = queryLen;

        return task;
    }
    
    void  svc_end()  {
        if (queryLib) {
            ff_closeLib( queryLib );
            queryLib=NULL;
        }
        // updating global counters
        qCount = lqCount;
        qResidues = lqResidues;
    }
    
private:
    char         * queryFile;
    char         * dbFile;
    char         * query;
    int            queryLen;
    FastaLib_ff  * queryLib;
    FastaLib_ff  * dbLib;
    char         * tofree; 
    char         * oldquery;
    unsigned int   lqCount;
    unsigned int   lqResidues;
};



int main( int argc, char * argv[] ){
	char * matrixFile = NULL, * queryFile = NULL, * dbFile = NULL;
#if defined(__SSE2__)
	SWType type = SSE2;
#elif defined(__ALTIVEC__)
    SWType type = ALTIVEC;
#else
	SWType type = SCALAR;
#endif
	Options options = {-12,-2,DBL_MAX};
#ifdef HAVE_SYSCONF_NPROCESSORS
	int threads = sysconf(_SC_NPROCESSORS_ONLN);
#else
	int threads = 1;
#endif

	for ( int i=1; i<argc; i++ ){
		if (argv[i][0]=='-'){
			switch( argv[i][1] ){
            case 'h':
                matrixFile = NULL;
                i = argc; break;
            case 's':
                type = SCALAR;
                break;
            case 't':
                options.threshold = atoi( argv[++i] );
                break;
            case 'i':
                options.gapOpen = atoi( argv[++i] );
                break;
            case 'e':
                options.gapExt = atoi( argv[++i] );
                break;
            case 'j':
                threads = atoi( argv[++i] );
                break;
            default:
                matrixFile = NULL;
                i = argc; break;
			}
		}else{
			if (matrixFile == NULL)
				matrixFile = argv[i];
			else if (queryFile == NULL)
				queryFile = argv[i];
			else if (dbFile == NULL)
				dbFile = argv[i];
			else{
				matrixFile = NULL;
				i = argc; break;
			}
		}
	}
	if ( matrixFile == NULL || queryFile == NULL || dbFile == NULL ){
		printf( "Usage: %s [-h] [-s] [-j num] [-i num] [-e num] [-t num] matrix query db\n", argv[0] );
		return 0;
	}
    
	SBMatrix matrix   = swps3_readSBMatrix( matrixFile );
    
    //ALLOCATOR_INIT();
    ALLOCATOR_INIT1();

    // worker's args
    args_t args;
    args.type    = type;
    args.options = options;
    args.matrix  = matrix;
    qCount=0;qResidues=0;dCount=0;dResidues=0;

	// create the farm object
	ff_farm<> farm(false, INPUT_CHANNEL_SLOTS);
    std::vector<ff_node *> w;
    for(int i=0;i<threads;++i) 
        w.push_back(new Worker(type,options, matrix));
    farm.add_workers(w);
    
	
	// create and add to the farm the emitter object
	Emitter E(queryFile,dbFile);
	farm.add_emitter(&E);
    
	// let's start
	if (farm.run_and_wait_end()<0) {
	    error("running farm\n");
	    return -1;
	}
    
    double time = farm.ffTime();
    
    fprintf(stderr,"%d[%d] x %d[%d]\n", qCount, qResidues, dCount, dResidues );	
    fprintf(stderr,"TIME (ms) = %lf  GCUPS= %lf\n", time, (dResidues/1e6)*(qResidues/time));
    farm.ffStats(std::cerr);
    fprintf(stderr,"\nDONE\n");
    ALLOCATOR_ST;            
	return 0;
}
