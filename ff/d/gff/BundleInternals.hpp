/*
 * bundleInternals.hpp
 *
 *  Created on: Nov 21, 2018
 *      Author: pvi
 */

#ifndef FF_D_GFF_BUNDLEINTERNALS_HPP_
#define FF_D_GFF_BUNDLEINTERNALS_HPP_

#include <gam.hpp>

#include "Logger.hpp"

namespace gff {

template <typename Comm>
struct BundleInternals {
	void source(gam::executor_id s) {
		for (auto communicator : commBundle)
			communicator.internals.source(s);
	}

	void destination(gam::executor_id d) {
		for (auto communicator : commBundle)
			communicator.internals.destination(d);
	}

	gam::executor_id in_cardinality() {
		gam::executor_id count = 0;
		for (auto communicator : commBundle)
			count += communicator.internals.in_cardinality();
		return count;
	}

	template<typename T, typename ... PolicyArgs>
	void put(const gam::public_ptr<T> &p, PolicyArgs&&... __a) {
		for (auto communicator : commBundle) {
			GFF_LOGLN_OS("COM put public=" << p);
			communicator.internals.put(p, std::forward<PolicyArgs>(__a)...);
		}
	}

	template<typename T>
	void broadcast(const gam::public_ptr<T> &p) {
		for (auto communicator : commBundle) {
			GFF_LOGLN_OS("COM broadcast public=" << p);
			communicator.internals.broadcast(p);
		}
	}

	vector<Comm> commBundle;

};

} /* namespace gam */

#endif /* FF_D_GFF_BUNDLEINTERNALS_HPP_ */
