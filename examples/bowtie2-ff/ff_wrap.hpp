/* 
 * ***************************************************************************  
 * **
 * **  Authors: Claudia Misale <misale@di.unito.it>
 * **             (http://alpha.di.unito.it/claudia-misale)
 * **           Marco Aldinucci <aldinuc@di.unito.it>
 * **             (http://alpha.di.unito.it/marco-aldinucci)
 * **   
 * **  Bowtie2 version: bowtie2-2.0.6
 * **             
 * **  FastFlow version: fastflow-2.0.4
 * **
 * **  Date  :  July 8, 2014
 * **
 * **  This file is part of the FastFlow examples. 
 * **  http://sourceforge.net/projects/mc-fastflow/
 * **
 * **  Licence: Fastflow is LGPLv3. Bowtie-ff is GPL. See licence file.
 * **
 * ** ************************************************************************
 ***/






#ifndef FF_WRAP
#define FF_WRAP
#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/allocator.hpp>
#include <ff/svector.hpp>

#define FFMALLOC(size)   (FFAllocator::instance()->malloc(size))
#define FFFREE(ptr) (FFAllocator::instance()->free(ptr))

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <stdexcept>
#if defined(_WIN32) && defined(FF)
#include <ff/platforms/platform.h>
#else
#include <getopt.h>
#include <pthread.h>
#endif
#include <math.h>
#include <utility>
#include <limits>
#include "alphabet.h"
#include "assert_helpers.h"
#include "endian_swap.h"
#include "bt2_idx.h"
#include "formats.h"
#include "sequence_io.h"
#include "tokenize.h"
#include "aln_sink.h"
#include "pat.h"
#include "threading.h"
#include "ds.h"
#include "aligner_metrics.h"
#include "sam.h"
#include "aligner_seed.h"
#include "aligner_seed_policy.h"
#include "aligner_driver.h"
#include "aligner_sw.h"
#include "aligner_sw_driver.h"
#include "aligner_cache.h"
#include "util.h"
#include "pe.h"
#include "simple_func.h"
#include "presets.h"
#include "opts.h"
#include "outq.h"
#include "aligner_seed2.h"
#include "metrics_utils.hpp"
#include "aligner_sw_common.h"

#ifdef FF_NUMA_ALLOC
#include <numa.h>
#endif

using namespace ff;
using namespace std;

#define INBUF_Q_SIZE 8

typedef struct {
	Read* bufa;
	Read* bufb;
	TReadId rdid;
	bool success;
	bool done;
	bool paired;
	// new entries
	OuterLoopMetrics* olm;
	SeedSearchMetrics* sdm;
	WalkMetrics* wlm;
	SwMetrics *swmSeed, *swmMate;
	ReportingMetrics* rpm;
	SSEMetrics* sseU8ExtendMet;
	SSEMetrics* sseU8MateMet;
	SSEMetrics* sseI16ExtendMet;
	SSEMetrics* sseI16MateMet;
	uint64_t* nbtfiltst; // TODO: find a new home for these
	uint64_t* nbtfiltsc; // TODO: find a new home for these
	uint64_t* nbtfiltdo; // TODO: find a new home for these
#ifdef FF_NUMA_ALLOC
	int sourceW;
	int mnode;
	int core;
#endif
} ff_task;

class EmitterPatSrc: public ff_node {
public:

	EmitterPatSrc(uint32_t ntask_, PairedPatternSource& pairedPatSrc_, int& outType_,
			int numthreads_, AlnSink& msink_, PerfMetrics& metrics_,
			ff_loadbalancer* const lb_, const svector<ff_node*>& workers_) {
		pairedPatSrc = &pairedPatSrc_;
		outType = outType_;
		maxReads = ntask_;
		rdid = 0;
		endid = 0;
		numtask = INBUF_Q_SIZE * numthreads_;
		msink = &msink_;
		metrics = &metrics_;
		count = 0;
		nworkers = numthreads_;
		lb = lb_;
		workers = workers_;
		elapsedTime = 0, msSum = 0;
	}

	int svc_init() {
		return 0;
	}
	void svc_end() {
	//	printf("%f\n",msSum);
	}

	void * svc(void *t) {
		success = false;
		done = false;
		paired = false;
		ff_task *alignTask = (ff_task*) t;
		//struct timeval t1, t2;


		if (alignTask == NULL) {
			for (int i = 0; i < INBUF_Q_SIZE; i++) {
				for (int j = 0; j < nworkers; j++) {
#ifdef FF_NUMA_ALLOC	
					int targetworker = workers[j]->get_my_id();
					int targetcore = threadMapper::instance()->getCoreId(lb->getTid(workers[j]));
					int targetmnode = (targetcore%32)/ 8; //nehalem
					//int targetmnode = (targetcore%16)/ 8; //sandybridge
					alignTask = (ff_task*) numa_alloc_onnode(sizeof(ff_task), targetmnode);
					alignTask->bufa = (Read*) numa_alloc_onnode(sizeof(Read), targetmnode);
					alignTask->bufb = (Read*) numa_alloc_onnode(sizeof(Read), targetmnode);
					void * p = numa_alloc_onnode(sizeof(OuterLoopMetrics), targetmnode);
					alignTask->olm = new (p) OuterLoopMetrics();

					p = numa_alloc_onnode(sizeof(SeedSearchMetrics), targetmnode);
					alignTask->sdm = new (p) SeedSearchMetrics();

					p = numa_alloc_onnode(sizeof(WalkMetrics), targetmnode);
					alignTask->wlm = new (p) WalkMetrics();

					p = numa_alloc_onnode(sizeof(SwMetrics), targetmnode);
					alignTask->swmSeed = new (p) SwMetrics();

					p = numa_alloc_onnode(sizeof(SwMetrics), targetmnode);
					alignTask->swmMate = new (p) SwMetrics();

					p = numa_alloc_onnode(sizeof(ReportingMetrics), targetmnode);
					alignTask->rpm = new (p) ReportingMetrics();

					p = numa_alloc_onnode(sizeof(SSEMetrics), targetmnode);
					alignTask->sseU8ExtendMet = new (p) SSEMetrics();

					p = numa_alloc_onnode(sizeof(SSEMetrics), targetmnode);
					alignTask->sseU8MateMet = new (p) SSEMetrics();

					p = numa_alloc_onnode(sizeof(SSEMetrics),targetmnode);
					alignTask->sseI16ExtendMet = new (p) SSEMetrics();

					p = numa_alloc_onnode(sizeof(SSEMetrics), targetmnode);
					alignTask->sseI16MateMet = new (p) SSEMetrics();

					p = numa_alloc_onnode(sizeof(uint64_t), targetmnode);
					alignTask->nbtfiltst = new (p) uint64_t(0);

					p = numa_alloc_onnode(sizeof(uint64_t), targetmnode);
					alignTask->nbtfiltsc = new (p) uint64_t(0);

					p = numa_alloc_onnode(sizeof(uint64_t), targetmnode);
					alignTask->nbtfiltdo = new (p) uint64_t(0);

					alignTask->sourceW = targetworker;
					alignTask->core = targetcore;
					alignTask->mnode = targetmnode;

					//move other
#else
					alignTask = (ff_task*) FFMALLOC(sizeof(ff_task));
					alignTask->bufa = (Read*) FFMALLOC(sizeof(Read));
					alignTask->bufb = (Read*) FFMALLOC(sizeof(Read));
					alignTask->olm = new OuterLoopMetrics();
					alignTask->sdm = new SeedSearchMetrics();
					alignTask->wlm = new WalkMetrics();
					alignTask->swmSeed = new SwMetrics();
					alignTask->swmMate = new SwMetrics();
					alignTask->rpm = new ReportingMetrics();
					alignTask->sseU8ExtendMet = new SSEMetrics();
					alignTask->sseU8MateMet = new SSEMetrics();
					alignTask->sseI16ExtendMet = new SSEMetrics();
					alignTask->sseI16MateMet = new SSEMetrics();
					alignTask->nbtfiltst = new uint64_t(0);
					alignTask->nbtfiltsc = new uint64_t(0);
					alignTask->nbtfiltdo = new uint64_t(0);
#endif
					//gettimeofday(&t1, NULL);
					pairedPatSrc->nextReadPair(*alignTask->bufa, *alignTask->bufb, rdid,
							endid, success, done, paired, outType != OUTPUT_SAM);
					//gettimeofday(&t2, NULL);

					//elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;     // sec to ms
					//elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
					//msSum += elapsedTime;

					if (!success && done) {
#ifdef FF_NUMA_ALLOC
						numa_free(alignTask->olm, sizeof(OuterLoopMetrics));
						numa_free(alignTask->sdm, sizeof(SeedSearchMetrics));
						numa_free(alignTask->wlm, sizeof(WalkMetrics));
						numa_free(alignTask->swmSeed, sizeof(SwMetrics));
						numa_free(alignTask->swmMate, sizeof(SwMetrics));
						numa_free(alignTask->rpm, sizeof(ReportingMetrics));
						numa_free(alignTask->sseU8ExtendMet,
								sizeof(SSEMetrics));
						numa_free(alignTask->sseU8MateMet, sizeof(SSEMetrics));
						numa_free(alignTask->sseI16ExtendMet,
								sizeof(SSEMetrics));
						numa_free(alignTask->sseI16MateMet, sizeof(SSEMetrics));
						numa_free(alignTask->nbtfiltsc, sizeof(uint64_t));
						numa_free(alignTask->nbtfiltst, sizeof(uint64_t));
						numa_free(alignTask->nbtfiltdo, sizeof(uint64_t));
						numa_free(alignTask->bufb, sizeof(Read));
						numa_free(alignTask->bufa, sizeof(Read));
						numa_free(alignTask, sizeof(ff_task));
#else
						delete alignTask->olm;
						delete alignTask->sdm;
						delete alignTask->wlm;
						delete alignTask->swmSeed;
						delete alignTask->swmMate;
						delete alignTask->rpm;
						delete alignTask->sseU8ExtendMet;
						delete alignTask->sseU8MateMet;
						delete alignTask->sseI16ExtendMet;
						delete alignTask->sseI16MateMet;
						delete alignTask->nbtfiltsc;
						delete alignTask->nbtfiltst;
						delete alignTask->nbtfiltdo;
						FFFREE(alignTask->bufa);
						FFFREE(alignTask->bufb);
						FFFREE(alignTask);
#endif
						return NULL;
					}

					if (rdid <(unsigned) maxReads || maxReads == -1) {
						count += 1;
						alignTask->done = done;
						alignTask->paired = paired;
						alignTask->rdid = rdid;
						alignTask->success = success;
#ifdef FF_NUMA_ALLOC
						bool res = lb->ff_send_out_to(alignTask, targetworker);
						if (!res) {
							printf(
									"ERROR: send failed - should never happen - task is lost - queue are too short\n");
						}
#else
						ff_send_out(alignTask);
#endif
					}
				}

				} // fine for
				return GO_ON;

			} else {
				count += 1;
				msink->mergeMetrics(*(alignTask->rpm));
				metrics->merge((alignTask->olm), (alignTask->sdm), (alignTask->wlm),
						(alignTask->swmSeed), (alignTask->swmMate),
						(alignTask->rpm), (alignTask->sseU8ExtendMet),
						(alignTask->sseU8MateMet), (alignTask->sseI16ExtendMet),
						(alignTask->sseI16MateMet), *(alignTask->nbtfiltst),
						*(alignTask->nbtfiltsc), *(alignTask->nbtfiltdo), false);

		//		gettimeofday(&t1, NULL);
				pairedPatSrc->nextReadPair(*alignTask->bufa, *alignTask->bufb, rdid,
						endid, success, done, paired, outType != OUTPUT_SAM);
		//		gettimeofday(&t2, NULL);

		//		elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;     // sec to ms
		//		elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
		//		msSum += elapsedTime;

				if (!success && done) {
#ifdef FF_NUMA_ALLOC
					numa_free(alignTask->olm, sizeof(OuterLoopMetrics));
					numa_free(alignTask->sdm, sizeof(SeedSearchMetrics));
					numa_free(alignTask->wlm, sizeof(WalkMetrics));
					numa_free(alignTask->swmSeed, sizeof(SwMetrics));
					numa_free(alignTask->swmMate, sizeof(SwMetrics));
					numa_free(alignTask->rpm, sizeof(ReportingMetrics));
					numa_free(alignTask->sseU8ExtendMet, sizeof(SSEMetrics));
					numa_free(alignTask->sseU8MateMet, sizeof(SSEMetrics));
					numa_free(alignTask->sseI16ExtendMet, sizeof(SSEMetrics));
					numa_free(alignTask->sseI16MateMet, sizeof(SSEMetrics));
					numa_free(alignTask->nbtfiltsc, sizeof(uint64_t));
					numa_free(alignTask->nbtfiltst, sizeof(uint64_t));
					numa_free(alignTask->nbtfiltdo, sizeof(uint64_t));
					numa_free(alignTask->bufb, sizeof(Read));
					numa_free(alignTask->bufa, sizeof(Read));
					numa_free(alignTask, sizeof(ff_task));
#else
					delete alignTask->olm;
					delete alignTask->sdm;
					delete alignTask->wlm;
					delete alignTask->swmSeed;
					delete alignTask->swmMate;
					delete alignTask->rpm;
					delete alignTask->sseU8ExtendMet;
					delete alignTask->sseU8MateMet;
					delete alignTask->sseI16ExtendMet;
					delete alignTask->sseI16MateMet;
					delete alignTask->nbtfiltsc;
					delete alignTask->nbtfiltst;
					delete alignTask->nbtfiltdo;
					FFFREE(alignTask->bufa);
					FFFREE(alignTask->bufb);
					FFFREE(alignTask);
#endif
					return NULL;
				}

				if ((rdid < (unsigned) maxReads || maxReads == -1)) {
					alignTask->olm->reset();
					alignTask->sdm->reset();
					alignTask->wlm->reset();
					alignTask->swmSeed->reset();
					alignTask->swmMate->reset();
					alignTask->rpm->reset();
					alignTask->sseU8ExtendMet->reset();
					alignTask->sseU8MateMet->reset();
					alignTask->sseI16ExtendMet->reset();
					alignTask->sseI16MateMet->reset();
					alignTask->done = done;
					alignTask->paired = paired;
					alignTask->rdid = rdid;
					alignTask->success = success;
#ifdef FF_NUMA_ALLOC
					bool res = lb->ff_send_out_to(alignTask, alignTask->sourceW);
					if (!res) {
						printf(
								"ERROR: send failed - should never happen - task is lost - queue are too short\n");
					}
					return GO_ON;
#else
					return (alignTask);
#endif

				}
			}
		return NULL;

	}

private:
	double elapsedTime;
	double msSum;
	int maxReads;
	PairedPatternSource* pairedPatSrc;
	int outType;
	bool success, done, paired;
	TReadId rdid;
	TReadId endid;
	uint32_t numtask;
	AlnSink* msink;
	PerfMetrics* metrics;
	long count;
	int nworkers;
	ff_loadbalancer* lb;
	svector<ff_node*> workers;
}
;

#endif
