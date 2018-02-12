/* ***************************************************************************
 *
 *  This file is part of FastFlow.
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  FastFlow is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with FastFlow. If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************
 */

/*
 * utils.hpp
 *
 *  Created on: Jun 13, 2017
 *      Author: drocco
 */

#ifndef FF_D_GFF_UTILS_HPP_
#define FF_D_GFF_UTILS_HPP_

/*
 * High resolution timers on top of C++ chrono
 */

#include <chrono>

namespace gff {
typedef std::chrono::high_resolution_clock::time_point time_point_t;
typedef std::chrono::duration<double> duration_t;

time_point_t hires_timer_ull() {
	return std::chrono::high_resolution_clock::now();
}

duration_t time_diff(time_point_t a, time_point_t b) {
	return std::chrono::duration_cast<duration_t>(b - a);
}

} /* namespace gff */

#endif /* FF_D_GFF_UTILS_HPP_ */
