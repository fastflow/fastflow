/* ***************************************************************************
 *
 *  This file is part of gam.
 *
 *  gam is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gam is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with gam. If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************
 */

/**
 * @file        Profiler.hpp
 * @brief       implements GFF Profiler class and shortcut macros
 * @author      Maurizio Drocco
 *
 * @ingroup     internals
 */
#ifndef FF_D_GFF_PROFILER_HPP_
#define FF_D_GFF_PROFILER_HPP_

#include <cassert>
#include <fstream>
#include <iostream>
#include <cstdarg>
#include <string>
#include <unistd.h>
using namespace std;

#include "utils.hpp"

/*
 * shortcuts
 */
#ifdef GFF_PROFILE
#define GFF_PROFLN (gff::Profiler::getProfiler()->log)
#define GFF_PROFLN_RAW (gff::Profiler::getProfiler()->log_raw)
#define GFF_PROFLN_OS(x) (gff::Profiler::getProfiler()->out_stream() << x << std::endl)
#define GFF_PROFILER_INIT (gff::Profiler::getProfiler()->init)
#define GFF_PROFILER_FINALIZE (gff::Profiler::getProfiler()->finalize)
#define GFF_PROFILER_TIMER(v) time_point_t v
#define GFF_PROFILER_HRT(v) (v) = hires_timer_ull()
#define GFF_PROFILER_SPAN(a,b) time_diff((a),(b)).count()
#define GFF_PROFILER_DURATION(v) duration_t v(0)
#define GFF_PROFILER_DURATION_ADD(v,a,b) (v) += time_diff((a),(b))
#define GFF_PROFILER_COND(s) s
#define GFF_PROFILER_BOOL(v,c) bool v = (c);
#define GFF_PROFILER_BOOL_COND(v,s) if(v) {s}
#else
#define GFF_PROFLN(...) {}
#define GFF_PROFLN_RAW(...) {}
#define GFF_PROFLN_OS(...) {}
#define GFF_PROFILER_INIT(...) {}
#define GFF_PROFILER_FINALIZE(...) {}
#define GFF_PROFILER_TIMER(...) {}
#define GFF_PROFILER_HRT(...) {}
#define GFF_PROFILER_SPAN(...) {}
#define GFF_PROFILER_DURATION(...) {}
#define GFF_PROFILER_DURATION_ADD(...) {}
#define GFF_PROFILER_COND(...) {}
#define GFF_PROFILER_BOOL(...) {}
#define GFF_PROFILER_BOOL_COND(...) {}
#endif

namespace gff {

class Profiler {
public:
	static Profiler *getProfiler() {
		static Profiler logger;
		return &logger;
	}

	void init(std::string fname, int id) {
		//print header message
		log("I am GFF node %d", id);
	}
	void finalize(int id) {
		//print footer message
		log("stop profiling GFF node %d", id);
	}

	/**
	 *   Variable Length Logger function
	 *   @param format string for the message to be logged.
	 */
	void log(const char * format, ...) {
		//print message
		va_start(args, format);
		vsprintf(sMessage, format, args);
		out_stream() << sMessage << std::endl;
		va_end(args);
	}

	/**
	 *   Variable Length Logger function
	 *   @param format string for the message to be logged.
	 */
	void log_raw(const char * format, ...) {
		//print message
		va_start(args, format);
		vsprintf(sMessage, format, args);
		out_stream() << sMessage << std::endl;
		va_end(args);
	}

	std::ostream &out_stream() {
		return cout << "[" << getTime() << "] ";
	}

private:
	char sMessage[256];
	va_list args;

	std::string getTime() {
		time_t now = time(0);
		if (now != -1) {
			std::string time = ctime(&now);
			time.pop_back();
			return time;
		}
		return "-";
	}

};

} /* namespace gff */

#endif /* FF_D_GFF_PROFILER_HPP_ */
