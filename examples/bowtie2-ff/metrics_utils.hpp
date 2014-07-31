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


#ifndef BT2FF_METRIC_UTILS
#define BT2FF_METRIC_UTILS

#include "aligner_swsse.h"

/**
 * Metrics for measuring the work done by the outer read alignment
 * loop.
 */
struct OuterLoopMetrics {

	OuterLoopMetrics() { reset(); MUTEX_INIT(lock); }

	/**
	 * Set all counters to 0.
	 */
	void reset() {
		reads = bases = srreads = srbases =
		freads = fbases = ureads = ubases = 0;
	}

	/**
	 * Sum the counters in m in with the conters in this object.  This
	 * is the only safe way to update an OuterLoopMetrics that's shared
	 * by multiple threads.
	 */
	void merge(
		const OuterLoopMetrics& m,
		bool getLock = false)
	{
#ifndef FF
		ThreadSafe ts(&lock, getLock);
#endif
		reads += m.reads;
		bases += m.bases;
		srreads += m.srreads;
		srbases += m.srbases;
		freads += m.freads;
		fbases += m.fbases;
		ureads += m.ureads;
		ubases += m.ubases;
	}

	uint64_t reads;   // total reads
	uint64_t bases;   // total bases
	uint64_t srreads; // same-read reads
	uint64_t srbases; // same-read bases
	uint64_t freads;  // filtered reads
	uint64_t fbases;  // filtered bases
	uint64_t ureads;  // unfiltered reads
	uint64_t ubases;  // unfiltered bases
	MUTEX_T lock;
};
/**
 * Collection of all relevant performance metrics when aligning in
 * multiseed mode.
 */
struct PerfMetrics {

	PerfMetrics() : first(true) { reset(); }

	/**
	 * Set all counters to 0.
	 */
	void reset() {
		olm.reset();
		sdm.reset();
		wlm.reset();
		swmSeed.reset();
		swmMate.reset();
		rpm.reset();
		dpSse8Seed.reset();   // 8-bit SSE seed extensions
		dpSse8Mate.reset();   // 8-bit SSE mate finds
		dpSse16Seed.reset();  // 16-bit SSE seed extensions
		dpSse16Mate.reset();  // 16-bit SSE mate finds
		nbtfiltst = 0;
		nbtfiltsc = 0;
		nbtfiltdo = 0;

		olmu.reset();
		sdmu.reset();
		wlmu.reset();
		swmuSeed.reset();
		swmuMate.reset();
		rpmu.reset();
		dpSse8uSeed.reset();  // 8-bit SSE seed extensions
		dpSse8uMate.reset();  // 8-bit SSE mate finds
		dpSse16uSeed.reset(); // 16-bit SSE seed extensions
		dpSse16uMate.reset(); // 16-bit SSE mate finds
		nbtfiltst_u = 0;
		nbtfiltsc_u = 0;
		nbtfiltdo_u = 0;
	}

	/**
	 * Merge a set of specific metrics into this object.
	 */
	void merge(
		const OuterLoopMetrics *ol,
		const SeedSearchMetrics *sd,
		const WalkMetrics *wl,
		const SwMetrics *swSeed,
		const SwMetrics *swMate,
		const ReportingMetrics *rm,
		const SSEMetrics *dpSse8Ex,
		const SSEMetrics *dpSse8Ma,
		const SSEMetrics *dpSse16Ex,
		const SSEMetrics *dpSse16Ma,
		uint64_t nbtfiltst_,
		uint64_t nbtfiltsc_,
		uint64_t nbtfiltdo_,
		bool getLock)
	{
#ifndef FF
		ThreadSafe ts(&lock, getLock);
#endif
		if(ol != NULL) {
			olmu.merge(*ol, false);
		}
		if(sd != NULL) {
			sdmu.merge(*sd, false);
		}
		if(wl != NULL) {
			wlmu.merge(*wl, false);
		}
		if(swSeed != NULL) {
			swmuSeed.merge(*swSeed, false);
		}
		if(swMate != NULL) {
			swmuMate.merge(*swMate, false);
		}
		if(rm != NULL) {
			rpmu.merge(*rm, false);
		}
		if(dpSse8Ex != NULL) {
			dpSse8uSeed.merge(*dpSse8Ex, false);
		}
		if(dpSse8Ma != NULL) {
			dpSse8uMate.merge(*dpSse8Ma, false);
		}
		if(dpSse16Ex != NULL) {
			dpSse16uSeed.merge(*dpSse16Ex, false);
		}
		if(dpSse16Ma != NULL) {
			dpSse16uMate.merge(*dpSse16Ma, false);
		}
		nbtfiltst_u += nbtfiltst_;
		nbtfiltsc_u += nbtfiltsc_;
		nbtfiltdo_u += nbtfiltdo_;
	}

	/**
	 * Reports a matrix of results, incl. column labels, to an OutFileBuf.
	 * Optionally also sends results to stderr (unbuffered).  Can optionally
	 * print a per-read record with the read name at the beginning.
	 */
	void reportInterval(
		OutFileBuf* o,        // file to send output to
		bool metricsStderr,   // additionally output to stderr?
		bool total,           // true -> report total, otherwise incremental
		bool sync,            //  synchronize output
		const BTString *name) // non-NULL name pointer if is per-read record
	{
#ifndef FF
		ThreadSafe ts(&lock, sync);
#endif
		ostringstream stderrSs;
		time_t curtime = time(0);
		char buf[1024];
		if(first) {
			const char *str =
				/*  1 */ "Time"           "\t"
				/*  2 */ "Read"           "\t"
				/*  3 */ "Base"           "\t"
				/*  4 */ "SameRead"       "\t"
				/*  5 */ "SameReadBase"   "\t"
				/*  6 */ "UnfilteredRead" "\t"
				/*  7 */ "UnfilteredBase" "\t"

				/*  8 */ "Paired"         "\t"
				/*  9 */ "Unpaired"       "\t"
				/* 10 */ "AlConUni"       "\t"
				/* 11 */ "AlConRep"       "\t"
				/* 12 */ "AlConFail"      "\t"
				/* 13 */ "AlDis"          "\t"
				/* 14 */ "AlConFailUni"   "\t"
				/* 15 */ "AlConFailRep"   "\t"
				/* 16 */ "AlConFailFail"  "\t"
				/* 17 */ "AlConRepUni"    "\t"
				/* 18 */ "AlConRepRep"    "\t"
				/* 19 */ "AlConRepFail"   "\t"
				/* 20 */ "AlUnpUni"       "\t"
				/* 21 */ "AlUnpRep"       "\t"
				/* 22 */ "AlUnpFail"      "\t"

				/* 23 */ "SeedSearch"     "\t"
				/* 24 */ "IntraSCacheHit" "\t"
				/* 25 */ "InterSCacheHit" "\t"
				/* 26 */ "OutOfMemory"    "\t"
				/* 27 */ "AlBWOp"         "\t"
				/* 28 */ "AlBWBranch"     "\t"
				/* 29 */ "ResBWOp"        "\t"
				/* 30 */ "ResBWBranch"    "\t"
				/* 31 */ "ResResolve"     "\t"
				/* 34 */ "ResReport"      "\t"
				/* 35 */ "RedundantSHit"  "\t"

				/* 36 */ "BestMinEdit0"   "\t"
				/* 37 */ "BestMinEdit1"   "\t"
				/* 38 */ "BestMinEdit2"   "\t"

				/* 39 */ "ExactAttempts"  "\t"
				/* 40 */ "ExactSucc"      "\t"
				/* 41 */ "ExactRanges"    "\t"
				/* 42 */ "ExactRows"      "\t"
				/* 43 */ "ExactOOMs"      "\t"

				/* 44 */ "1mmAttempts"    "\t"
				/* 45 */ "1mmSucc"        "\t"
				/* 46 */ "1mmRanges"      "\t"
				/* 47 */ "1mmRows"        "\t"
				/* 48 */ "1mmOOMs"        "\t"

				/* 49 */ "UngappedSucc"   "\t"
				/* 50 */ "UngappedFail"   "\t"
				/* 51 */ "UngappedNoDec"  "\t"

				/* 52 */ "DPExLt10Gaps"   "\t"
				/* 53 */ "DPExLt5Gaps"    "\t"
				/* 54 */ "DPExLt3Gaps"    "\t"

				/* 55 */ "DPMateLt10Gaps" "\t"
				/* 56 */ "DPMateLt5Gaps"  "\t"
				/* 57 */ "DPMateLt3Gaps"  "\t"

				/* 58 */ "DP16ExDps"      "\t"
				/* 59 */ "DP16ExDpSat"    "\t"
				/* 60 */ "DP16ExDpFail"   "\t"
				/* 61 */ "DP16ExDpSucc"   "\t"
				/* 62 */ "DP16ExCol"      "\t"
				/* 63 */ "DP16ExCell"     "\t"
				/* 64 */ "DP16ExInner"    "\t"
				/* 65 */ "DP16ExFixup"    "\t"
				/* 66 */ "DP16ExGathSol"  "\t"
				/* 67 */ "DP16ExBt"       "\t"
				/* 68 */ "DP16ExBtFail"   "\t"
				/* 69 */ "DP16ExBtSucc"   "\t"
				/* 70 */ "DP16ExBtCell"   "\t"
				/* 71 */ "DP16ExCoreRej"  "\t"
				/* 72 */ "DP16ExNRej"     "\t"

				/* 73 */ "DP8ExDps"       "\t"
				/* 74 */ "DP8ExDpSat"     "\t"
				/* 75 */ "DP8ExDpFail"    "\t"
				/* 76 */ "DP8ExDpSucc"    "\t"
				/* 77 */ "DP8ExCol"       "\t"
				/* 78 */ "DP8ExCell"      "\t"
				/* 79 */ "DP8ExInner"     "\t"
				/* 80 */ "DP8ExFixup"     "\t"
				/* 81 */ "DP8ExGathSol"   "\t"
				/* 82 */ "DP8ExBt"        "\t"
				/* 83 */ "DP8ExBtFail"    "\t"
				/* 84 */ "DP8ExBtSucc"    "\t"
				/* 85 */ "DP8ExBtCell"    "\t"
				/* 86 */ "DP8ExCoreRej"   "\t"
				/* 87 */ "DP8ExNRej"      "\t"

				/* 88 */ "DP16MateDps"     "\t"
				/* 89 */ "DP16MateDpSat"   "\t"
				/* 90 */ "DP16MateDpFail"  "\t"
				/* 91 */ "DP16MateDpSucc"  "\t"
				/* 92 */ "DP16MateCol"     "\t"
				/* 93 */ "DP16MateCell"    "\t"
				/* 94 */ "DP16MateInner"   "\t"
				/* 95 */ "DP16MateFixup"   "\t"
				/* 96 */ "DP16MateGathSol" "\t"
				/* 97 */ "DP16MateBt"      "\t"
				/* 98 */ "DP16MateBtFail"  "\t"
				/* 99 */ "DP16MateBtSucc"  "\t"
				/* 100 */ "DP16MateBtCell"  "\t"
				/* 101 */ "DP16MateCoreRej" "\t"
				/* 102 */ "DP16MateNRej"    "\t"

				/* 103 */ "DP8MateDps"     "\t"
				/* 104 */ "DP8MateDpSat"   "\t"
				/* 105 */ "DP8MateDpFail"  "\t"
				/* 106 */ "DP8MateDpSucc"  "\t"
				/* 107 */ "DP8MateCol"     "\t"
				/* 108 */ "DP8MateCell"    "\t"
				/* 109 */ "DP8MateInner"   "\t"
				/* 110 */ "DP8MateFixup"   "\t"
				/* 111 */ "DP8MateGathSol" "\t"
				/* 112 */ "DP8MateBt"      "\t"
				/* 113 */ "DP8MateBtFail"  "\t"
				/* 114 */ "DP8MateBtSucc"  "\t"
				/* 115 */ "DP8MateBtCell"  "\t"
				/* 116 */ "DP8MateCoreRej" "\t"
				/* 117 */ "DP8MateNRej"    "\t"

				/* 118 */ "DPBtFiltStart"  "\t"
				/* 119 */ "DPBtFiltScore"  "\t"
				/* 120 */ "DpBtFiltDom"    "\t"

				/* 121 */ "MemPeak"        "\t"
				/* 122 */ "UncatMemPeak"   "\t" // 0
				/* 123 */ "EbwtMemPeak"    "\t" // EBWT_CAT
				/* 124 */ "CacheMemPeak"   "\t" // CA_CAT
				/* 125 */ "ResolveMemPeak" "\t" // GW_CAT
				/* 126 */ "AlignMemPeak"   "\t" // AL_CAT
				/* 127 */ "DPMemPeak"      "\t" // DP_CAT
				/* 128 */ "MiscMemPeak"    "\t" // MISC_CAT
				/* 129 */ "DebugMemPeak"   "\t" // DEBUG_CAT

				"\n";

			if(name != NULL) {
				if(o != NULL) o->writeChars("Name\t");
				if(metricsStderr) stderrSs << "Name\t";
			}

			if(o != NULL) o->writeChars(str);
			if(metricsStderr) stderrSs << str;
			first = false;
		}

		if(total) mergeIncrementals();

		// 0. Read name, if needed
		if(name != NULL) {
			if(o != NULL) {
				o->writeChars(name->toZBuf());
				o->write('\t');
			}
			if(metricsStderr) {
				stderrSs << (*name) << '\t';
			}
		}

		// 1. Current time in secs
		itoa10<time_t>(curtime, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		const OuterLoopMetrics& ol = total ? olm : olmu;

		// 2. Reads
		itoa10<uint64_t>(ol.reads, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 3. Bases
		itoa10<uint64_t>(ol.bases, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 4. Same-read reads
		itoa10<uint64_t>(ol.srreads, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 5. Same-read bases
		itoa10<uint64_t>(ol.srbases, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 6. Unfiltered reads
		itoa10<uint64_t>(ol.ureads, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 7. Unfiltered bases
		itoa10<uint64_t>(ol.ubases, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		const ReportingMetrics& rp = total ? rpm : rpmu;

		// 8. Paired reads
		itoa10<uint64_t>(rp.npaired, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 9. Unpaired reads
		itoa10<uint64_t>(rp.nunpaired, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 10. Pairs with unique concordant alignments
		itoa10<uint64_t>(rp.nconcord_uni, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 11. Pairs with repetitive concordant alignments
		itoa10<uint64_t>(rp.nconcord_rep, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 12. Pairs with 0 concordant alignments
		itoa10<uint64_t>(rp.nconcord_0, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 13. Pairs with 1 discordant alignment
		itoa10<uint64_t>(rp.ndiscord, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 14. Mates from unaligned pairs that align uniquely
		itoa10<uint64_t>(rp.nunp_0_uni, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 15. Mates from unaligned pairs that align repetitively
		itoa10<uint64_t>(rp.nunp_0_rep, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 16. Mates from unaligned pairs that fail to align
		itoa10<uint64_t>(rp.nunp_0_0, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 17. Mates from repetitive pairs that align uniquely
		itoa10<uint64_t>(rp.nunp_rep_uni, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 18. Mates from repetitive pairs that align repetitively
		itoa10<uint64_t>(rp.nunp_rep_rep, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 19. Mates from repetitive pairs that fail to align
		itoa10<uint64_t>(rp.nunp_rep_0, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 20. Unpaired reads that align uniquely
		itoa10<uint64_t>(rp.nunp_uni, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 21. Unpaired reads that align repetitively
		itoa10<uint64_t>(rp.nunp_rep, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 22. Unpaired reads that fail to align
		itoa10<uint64_t>(rp.nunp_0, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		const SeedSearchMetrics& sd = total ? sdm : sdmu;

		// 23. Seed searches
		itoa10<uint64_t>(sd.seedsearch, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 24. Hits in 'current' cache
		itoa10<uint64_t>(sd.intrahit, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 25. Hits in 'local' cache
		itoa10<uint64_t>(sd.interhit, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 26. Out of memory
		itoa10<uint64_t>(sd.ooms, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 27. Burrows-Wheeler ops in aligner
		itoa10<uint64_t>(sd.bwops, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 28. Burrows-Wheeler branches (edits) in aligner
		itoa10<uint64_t>(sd.bweds, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		const WalkMetrics& wl = total ? wlm : wlmu;

		// 29. Burrows-Wheeler ops in resolver
		itoa10<uint64_t>(wl.bwops, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 30. Burrows-Wheeler branches in resolver
		itoa10<uint64_t>(wl.branches, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 31. Burrows-Wheeler offset resolutions
		itoa10<uint64_t>(wl.resolves, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 34. Offset reports
		itoa10<uint64_t>(wl.reports, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		// 35. Redundant seed hit
		itoa10<uint64_t>(total ? swmSeed.rshit : swmuSeed.rshit, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		// 36. # times the best (out of fw/rc) minimum # edits was 0
		itoa10<uint64_t>(total ? sdm.bestmin0 : sdmu.bestmin0, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 37. # times the best (out of fw/rc) minimum # edits was 1
		itoa10<uint64_t>(total ? sdm.bestmin1 : sdmu.bestmin1, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 38. # times the best (out of fw/rc) minimum # edits was 2
		itoa10<uint64_t>(total ? sdm.bestmin2 : sdmu.bestmin2, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		// 39. Exact aligner attempts
		itoa10<uint64_t>(total ? swmSeed.exatts : swmuSeed.exatts, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 40. Exact aligner successes
		itoa10<uint64_t>(total ? swmSeed.exsucc : swmuSeed.exsucc, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 41. Exact aligner ranges
		itoa10<uint64_t>(total ? swmSeed.exranges : swmuSeed.exranges, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 42. Exact aligner rows
		itoa10<uint64_t>(total ? swmSeed.exrows : swmuSeed.exrows, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 43. Exact aligner OOMs
		itoa10<uint64_t>(total ? swmSeed.exooms : swmuSeed.exooms, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		// 44. 1mm aligner attempts
		itoa10<uint64_t>(total ? swmSeed.mm1atts : swmuSeed.mm1atts, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 45. 1mm aligner successes
		itoa10<uint64_t>(total ? swmSeed.mm1succ : swmuSeed.mm1succ, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 46. 1mm aligner ranges
		itoa10<uint64_t>(total ? swmSeed.mm1ranges : swmuSeed.mm1ranges, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 47. 1mm aligner rows
		itoa10<uint64_t>(total ? swmSeed.mm1rows : swmuSeed.mm1rows, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 48. 1mm aligner OOMs
		itoa10<uint64_t>(total ? swmSeed.mm1ooms : swmuSeed.mm1ooms, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		// 49 Ungapped aligner success
		itoa10<uint64_t>(total ? swmSeed.ungapsucc : swmuSeed.ungapsucc, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 50. Ungapped aligner fail
		itoa10<uint64_t>(total ? swmSeed.ungapfail : swmuSeed.ungapfail, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 51. Ungapped aligner no decision
		itoa10<uint64_t>(total ? swmSeed.ungapnodec : swmuSeed.ungapnodec, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		// 52. # seed-extend DPs with < 10 gaps
		itoa10<uint64_t>(total ? swmSeed.sws10 : swmuSeed.sws10, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 53. # seed-extend DPs with < 5 gaps
		itoa10<uint64_t>(total ? swmSeed.sws5 : swmuSeed.sws5, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 54. # seed-extend DPs with < 3 gaps
		itoa10<uint64_t>(total ? swmSeed.sws3 : swmuSeed.sws3, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		// 55. # seed-extend DPs with < 10 gaps
		itoa10<uint64_t>(total ? swmMate.sws10 : swmuMate.sws10, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 56. # seed-extend DPs with < 5 gaps
		itoa10<uint64_t>(total ? swmMate.sws5 : swmuMate.sws5, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 57. # seed-extend DPs with < 3 gaps
		itoa10<uint64_t>(total ? swmMate.sws3 : swmuMate.sws3, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		const SSEMetrics& dpSse16s = total ? dpSse16Seed : dpSse16uSeed;

		// 58. 16-bit SSE seed-extend DPs tried
		itoa10<uint64_t>(dpSse16s.dp, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 59. 16-bit SSE seed-extend DPs saturated
		itoa10<uint64_t>(dpSse16s.dpsat, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 60. 16-bit SSE seed-extend DPs failed
		itoa10<uint64_t>(dpSse16s.dpfail, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 61. 16-bit SSE seed-extend DPs succeeded
		itoa10<uint64_t>(dpSse16s.dpsucc, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 62. 16-bit SSE seed-extend DP columns completed
		itoa10<uint64_t>(dpSse16s.col, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 63. 16-bit SSE seed-extend DP cells completed
		itoa10<uint64_t>(dpSse16s.cell, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 64. 16-bit SSE seed-extend DP inner loop iters completed
		itoa10<uint64_t>(dpSse16s.inner, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 65. 16-bit SSE seed-extend DP fixup loop iters completed
		itoa10<uint64_t>(dpSse16s.fixup, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 66. 16-bit SSE seed-extend DP gather, cells with potential solutions
		itoa10<uint64_t>(dpSse16s.gathsol, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 67. 16-bit SSE seed-extend DP backtrace attempts
		itoa10<uint64_t>(dpSse16s.bt, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 68. 16-bit SSE seed-extend DP failed backtrace attempts
		itoa10<uint64_t>(dpSse16s.btfail, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 69. 16-bit SSE seed-extend DP succesful backtrace attempts
		itoa10<uint64_t>(dpSse16s.btsucc, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 70. 16-bit SSE seed-extend DP backtrace cells
		itoa10<uint64_t>(dpSse16s.btcell, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 71. 16-bit SSE seed-extend DP core-diag rejections
		itoa10<uint64_t>(dpSse16s.corerej, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 72. 16-bit SSE seed-extend DP N rejections
		itoa10<uint64_t>(dpSse16s.nrej, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		const SSEMetrics& dpSse8s = total ? dpSse8Seed : dpSse8uSeed;

		// 73. 8-bit SSE seed-extend DPs tried
		itoa10<uint64_t>(dpSse8s.dp, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 74. 8-bit SSE seed-extend DPs saturated
		itoa10<uint64_t>(dpSse8s.dpsat, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 75. 8-bit SSE seed-extend DPs failed
		itoa10<uint64_t>(dpSse8s.dpfail, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 76. 8-bit SSE seed-extend DPs succeeded
		itoa10<uint64_t>(dpSse8s.dpsucc, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 77. 8-bit SSE seed-extend DP columns completed
		itoa10<uint64_t>(dpSse8s.col, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 78. 8-bit SSE seed-extend DP cells completed
		itoa10<uint64_t>(dpSse8s.cell, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 79. 8-bit SSE seed-extend DP inner loop iters completed
		itoa10<uint64_t>(dpSse8s.inner, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 80. 8-bit SSE seed-extend DP fixup loop iters completed
		itoa10<uint64_t>(dpSse8s.fixup, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 81. 16-bit SSE seed-extend DP gather, cells with potential solutions
		itoa10<uint64_t>(dpSse8s.gathsol, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 82. 16-bit SSE seed-extend DP backtrace attempts
		itoa10<uint64_t>(dpSse8s.bt, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 83. 16-bit SSE seed-extend DP failed backtrace attempts
		itoa10<uint64_t>(dpSse8s.btfail, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 84. 16-bit SSE seed-extend DP succesful backtrace attempts
		itoa10<uint64_t>(dpSse8s.btsucc, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 85. 16-bit SSE seed-extend DP backtrace cells
		itoa10<uint64_t>(dpSse8s.btcell, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 86. 16-bit SSE seed-extend DP core-diag rejections
		itoa10<uint64_t>(dpSse8s.corerej, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 87. 16-bit SSE seed-extend DP N rejections
		itoa10<uint64_t>(dpSse8s.nrej, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		const SSEMetrics& dpSse16m = total ? dpSse16Mate : dpSse16uMate;

		// 88. 16-bit SSE mate-finding DPs tried
		itoa10<uint64_t>(dpSse16m.dp, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 89. 16-bit SSE mate-finding DPs saturated
		itoa10<uint64_t>(dpSse16m.dpsat, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 90. 16-bit SSE mate-finding DPs failed
		itoa10<uint64_t>(dpSse16m.dpfail, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 91. 16-bit SSE mate-finding DPs succeeded
		itoa10<uint64_t>(dpSse16m.dpsucc, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 92. 16-bit SSE mate-finding DP columns completed
		itoa10<uint64_t>(dpSse16m.col, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 93. 16-bit SSE mate-finding DP cells completed
		itoa10<uint64_t>(dpSse16m.cell, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 94. 16-bit SSE mate-finding DP inner loop iters completed
		itoa10<uint64_t>(dpSse16m.inner, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 95. 16-bit SSE mate-finding DP fixup loop iters completed
		itoa10<uint64_t>(dpSse16m.fixup, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 96. 16-bit SSE mate-finding DP gather, cells with potential solutions
		itoa10<uint64_t>(dpSse16m.gathsol, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 97. 16-bit SSE mate-finding DP backtrace attempts
		itoa10<uint64_t>(dpSse16m.bt, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 98. 16-bit SSE mate-finding DP failed backtrace attempts
		itoa10<uint64_t>(dpSse16m.btfail, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 99. 16-bit SSE mate-finding DP succesful backtrace attempts
		itoa10<uint64_t>(dpSse16m.btsucc, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 100. 16-bit SSE mate-finding DP backtrace cells
		itoa10<uint64_t>(dpSse16m.btcell, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 101. 16-bit SSE mate-finding DP core-diag rejections
		itoa10<uint64_t>(dpSse16m.corerej, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 102. 16-bit SSE mate-finding DP N rejections
		itoa10<uint64_t>(dpSse16m.nrej, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		const SSEMetrics& dpSse8m = total ? dpSse8Mate : dpSse8uMate;

		// 103. 8-bit SSE mate-finding DPs tried
		itoa10<uint64_t>(dpSse8m.dp, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 104. 8-bit SSE mate-finding DPs saturated
		itoa10<uint64_t>(dpSse8m.dpsat, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 105. 8-bit SSE mate-finding DPs failed
		itoa10<uint64_t>(dpSse8m.dpfail, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 106. 8-bit SSE mate-finding DPs succeeded
		itoa10<uint64_t>(dpSse8m.dpsucc, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 107. 8-bit SSE mate-finding DP columns completed
		itoa10<uint64_t>(dpSse8m.col, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 108. 8-bit SSE mate-finding DP cells completed
		itoa10<uint64_t>(dpSse8m.cell, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 109. 8-bit SSE mate-finding DP inner loop iters completed
		itoa10<uint64_t>(dpSse8m.inner, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 110. 8-bit SSE mate-finding DP fixup loop iters completed
		itoa10<uint64_t>(dpSse8m.fixup, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 111. 16-bit SSE mate-finding DP gather, cells with potential solutions
		itoa10<uint64_t>(dpSse8m.gathsol, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 112. 16-bit SSE mate-finding DP backtrace attempts
		itoa10<uint64_t>(dpSse8m.bt, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 113. 16-bit SSE mate-finding DP failed backtrace attempts
		itoa10<uint64_t>(dpSse8m.btfail, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 114. 16-bit SSE mate-finding DP succesful backtrace attempts
		itoa10<uint64_t>(dpSse8m.btsucc, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 115. 16-bit SSE mate-finding DP backtrace cells
		itoa10<uint64_t>(dpSse8m.btcell, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 116. 16-bit SSE mate-finding DP core rejections
		itoa10<uint64_t>(dpSse8m.corerej, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 117. 16-bit SSE mate-finding N rejections
		itoa10<uint64_t>(dpSse8m.nrej, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		// 118. Backtrace candidates filtered due to starting cell
		itoa10<uint64_t>(total ? nbtfiltst : nbtfiltst_u, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 119. Backtrace candidates filtered due to low score
		itoa10<uint64_t>(total ? nbtfiltsc : nbtfiltsc_u, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 120. Backtrace candidates filtered due to domination
		itoa10<uint64_t>(total ? nbtfiltdo : nbtfiltdo_u, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }

		// 121. Overall memory peak
		itoa10<size_t>(gMemTally.peak() >> 20, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 122. Uncategorized memory peak
		itoa10<size_t>(gMemTally.peak(0) >> 20, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 123. Ebwt memory peak
		itoa10<size_t>(gMemTally.peak(EBWT_CAT) >> 20, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 124. Cache memory peak
		itoa10<size_t>(gMemTally.peak(CA_CAT) >> 20, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 125. Resolver memory peak
		itoa10<size_t>(gMemTally.peak(GW_CAT) >> 20, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 126. Seed aligner memory peak
		itoa10<size_t>(gMemTally.peak(AL_CAT) >> 20, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 127. Dynamic programming aligner memory peak
		itoa10<size_t>(gMemTally.peak(DP_CAT) >> 20, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 128. Miscellaneous memory peak
		itoa10<size_t>(gMemTally.peak(MISC_CAT) >> 20, buf);
		if(metricsStderr) stderrSs << buf << '\t';
		if(o != NULL) { o->writeChars(buf); o->write('\t'); }
		// 129. Debug memory peak
		itoa10<size_t>(gMemTally.peak(DEBUG_CAT) >> 20, buf);
		if(metricsStderr) stderrSs << buf;
		if(o != NULL) { o->writeChars(buf); }

		if(o != NULL) { o->write('\n'); }
		if(metricsStderr) cerr << stderrSs.str().c_str() << endl;
		if(!total) mergeIncrementals();
	}

	void mergeIncrementals() {
		olm.merge(olmu, false);
		sdm.merge(sdmu, false);
		wlm.merge(wlmu, false);
		swmSeed.merge(swmuSeed, false);
		swmMate.merge(swmuMate, false);
		dpSse8Seed.merge(dpSse8uSeed, false);
		dpSse8Mate.merge(dpSse8uMate, false);
		dpSse16Seed.merge(dpSse16uSeed, false);
		dpSse16Mate.merge(dpSse16uMate, false);
		nbtfiltst_u += nbtfiltst;
		nbtfiltsc_u += nbtfiltsc;
		nbtfiltdo_u += nbtfiltdo;

		olmu.reset();
		sdmu.reset();
		wlmu.reset();
		swmuSeed.reset();
		swmuMate.reset();
		rpmu.reset();
		dpSse8uSeed.reset();
		dpSse8uMate.reset();
		dpSse16uSeed.reset();
		dpSse16uMate.reset();
		nbtfiltst_u = 0;
		nbtfiltsc_u = 0;
		nbtfiltdo_u = 0;
	}

	// Total over the whole job
	OuterLoopMetrics  olm;   // overall metrics
	SeedSearchMetrics sdm;   // metrics related to seed alignment
	WalkMetrics       wlm;   // metrics related to walking left (i.e. resolving reference offsets)
	SwMetrics         swmSeed;  // metrics related to DP seed-extend alignment
	SwMetrics         swmMate;  // metrics related to DP mate-finding alignment
	ReportingMetrics  rpm;   // metrics related to reporting
	SSEMetrics        dpSse8Seed;  // 8-bit SSE seed extensions
	SSEMetrics        dpSse8Mate;    // 8-bit SSE mate finds
	SSEMetrics        dpSse16Seed; // 16-bit SSE seed extensions
	SSEMetrics        dpSse16Mate;   // 16-bit SSE mate finds
	uint64_t          nbtfiltst;
	uint64_t          nbtfiltsc;
	uint64_t          nbtfiltdo;

	// Just since the last update
	OuterLoopMetrics  olmu;  // overall metrics
	SeedSearchMetrics sdmu;  // metrics related to seed alignment
	WalkMetrics       wlmu;  // metrics related to walking left (i.e. resolving reference offsets)
	SwMetrics         swmuSeed; // metrics related to DP seed-extend alignment
	SwMetrics         swmuMate; // metrics related to DP mate-finding alignment
	ReportingMetrics  rpmu;  // metrics related to reporting
	SSEMetrics        dpSse8uSeed;  // 8-bit SSE seed extensions
	SSEMetrics        dpSse8uMate;  // 8-bit SSE mate finds
	SSEMetrics        dpSse16uSeed; // 16-bit SSE seed extensions
	SSEMetrics        dpSse16uMate; // 16-bit SSE mate finds
	uint64_t          nbtfiltst_u;
	uint64_t          nbtfiltsc_u;
	uint64_t          nbtfiltdo_u;

	MUTEX_T           lock;  // lock for when one ob
	bool              first; // yet to print first line?
	time_t            lastElapsed; // used in reportInterval to measure time since last call
};

#endif // BT2FF_METRIC_UTILS
