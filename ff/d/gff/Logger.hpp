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
 * @file        Logger.hpp
 * @brief       implements GFF Logger class and shortcut macros
 * @author      Maurizio Drocco
 *
 * @ingroup     internals
 */
#ifndef FF_D_GFF_LOGGER_HPP_
#define FF_D_GFF_LOGGER_HPP_

#include <cassert>
#include <fstream>
#include <iostream>
#include <cstdarg>
#include <string>
#include <unistd.h>
using namespace std;

/*
 * shortcuts
 */
#ifdef GFF_LOG
#define GFF_LOGLN (gff::Logger::getLogger()->log)
#define GFF_LOGLN_OS(x) (gff::Logger::getLogger()->out_stream() << x << std::endl)
#define GFF_LOGGER_INIT (gff::Logger::getLogger()->init)
#define GFF_LOGGER_FINALIZE (gff::Logger::getLogger()->finalize)
#else
#define GFF_LOGLN(...) {}
#define GFF_LOGLN_OS(...) {}
#define GFF_LOGGER_INIT(...) {}
#define GFF_LOGGER_FINALIZE(...) {}
#endif

namespace gff {

class Logger {
public:
	static Logger *getLogger() {
		static Logger logger;
		return &logger;
	}

	void init(std::string fname, int id) {
		//print header message
		log("I am GFF node %d", id);
	}
	void finalize(int id) {
		//print footer message
		log("stop logging GFF node %d", id);
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

} /* namespace gam */

#endif /* FF_D_GFF_LOGGER_HPP_ */
