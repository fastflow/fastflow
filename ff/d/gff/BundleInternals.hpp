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
class BundleInternals {
public:

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
		for (auto communicator : commBundle)
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

	vector<Comm> commBundle;

};

} /* namespace gam */

#endif /* FF_D_GFF_BUNDLEINTERNALS_HPP_ */
