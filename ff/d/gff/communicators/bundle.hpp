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
 *
 * @file        builtin.hpp
 * @author      Paolo Viviani
 *
 */
#ifndef FF_D_GFF_COMMUNICATORS_BUNDLE_HPP_
#define FF_D_GFF_COMMUNICATORS_BUNDLE_HPP_

#include <gam.hpp>

#include "../defs.hpp"
#include "../BundleInternals.hpp"

namespace gff {

template <typename Comm>
class OutBundleBroadcast {
public:
	/*
	 * In case the broadcast is not suitable, it is possible to point to specific communicators that are exposed
	 * as internals.get(id)
	 */
	template<typename T>
	void emit(const gam::public_ptr<T> &p) {
		for (auto communicator : internals.commBundle)
			communicator.emit(p);
	}

	Comm get(const gam::executor_id &id) {
		return internals.commBundle.at(id);
	}

	BundleInternals<Comm> internals;
};

} // namespce gff

#endif /* FF_D_GFF_COMMUNICATORS_BUNDLE_HPP_ */
