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
 * @file Communicator.hpp
 * @author Maurizio Drocco
 * @date Apr 11, 2017
 * @brief implements CommunicatorInternals class
 *
 */

#ifndef FF_D_GFF_COMMUNICATORINTERNALS_HPP_
#define FF_D_GFF_COMMUNICATORINTERNALS_HPP_

#include <vector>
#include <algorithm>

#include <gam.hpp>

#include "Logger.hpp"

namespace gff {

/**
 * @brief Passive Communicator
 */
template<typename PushDispatcher, typename PullDispatcher>
class CommunicatorInternals {
public:
	void source(gam::executor_id s) {
		input.push_back(s);
		std::sort(input.begin(), input.end());
	}

	void destination(gam::executor_id d) {
		output.push_back(d);
		std::sort(output.begin(), output.end());
	}

	gam::executor_id in_cardinality() const {
		return input.size();
	}

	gam::executor_id out_cardinality() const {
		return output.size();
	}

	template<typename T, typename ... PolicyArgs>
	void put(const gam::public_ptr<T> &p, PolicyArgs&&... __a) {
		GFF_LOGLN_OS("COM put public=" << p);
		push_dispatcher.put(output, p, std::forward<PolicyArgs>(__a)...);
	}

	template<typename T, typename ... PolicyArgs>
	void put(gam::private_ptr<T> &&p, PolicyArgs&&... __a) {
		GFF_LOGLN_OS("COM put private=" << p);
		push_dispatcher.put(output, std::move(p),
				std::forward<PolicyArgs>(__a)...);
	}

	template<typename ptr_t>
	ptr_t get() {
		ptr_t res;
		pull_dispatcher.get(input, res);
		GFF_LOGLN_OS("COM got pointer=" << res);
		return std::move(res);
	}

	template<typename T>
	void broadcast(const gam::public_ptr<T> &p) {
		GFF_LOGLN_OS("COM broadcast public=" << p);
		push_dispatcher.broadcast(output, p);
	}

	template<typename T>
	void broadcast(gam::private_ptr<T> &&p) {
		GFF_LOGLN_OS("COM broadcast private=" << p);
		push_dispatcher.broadcast(output, std::move(p));
	}

private:
	PushDispatcher push_dispatcher;
	PullDispatcher pull_dispatcher;
	std::vector<gam::executor_id> input, output;
};

} /* namespace gam */

#endif /* FF_D_GFF_COMMUNICATORINTERNALS_HPP_ */
