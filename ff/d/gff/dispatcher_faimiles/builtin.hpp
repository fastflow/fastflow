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
 * @author      Maurizio Drocco
 * 
 */
#ifndef FF_D_GFF_DISPATCHER_FAIMILES_BUILTIN_HPP_
#define FF_D_GFF_DISPATCHER_FAIMILES_BUILTIN_HPP_

namespace gff {

#include <vector>

#include <gam.hpp>

/**
 * Nondeterminate Merge (singleton) family.
 */
class NDMerge {
public:
	template<typename T>
	void get(const std::vector<gam::executor_id> &s, //
			gam::public_ptr<T> &p) {
		p = gam::pull_public<T>();

	}

	template<typename T>
	void get(const std::vector<gam::executor_id> &s, //
			gam::private_ptr<T> &p) {
		p = gam::pull_private<T>();
	}
};

/**
 * Merge family.
 */
template<typename Policy>
class Merge {
public:
	template<typename T>
	using ptr_t = gam::private_ptr<T>;

	template<typename T>
	void get(const std::vector<gam::executor_id> &s, //
			gam::public_ptr<T> &p) {
		p = gam::pull_public<T>(coll(s));
	}

	template<typename T>
	void get(const std::vector<gam::executor_id> &s, //
			gam::private_ptr<T> &p) {
		p = gam::pull_private<T>(coll(s));
	}

private:
	Policy coll;
};

/**
 * Switch family.
 */
template<typename Policy>
class Switch {
public:
	template<typename T, typename ... PolicyArgs>
	void put(const std::vector<gam::executor_id> &d, //
			const gam::public_ptr<T> &p, //
			PolicyArgs&&... __a) {
		p.push(dist(d, std::forward<PolicyArgs>(__a)...));
	}

	template<typename T, typename ... PolicyArgs>
	void put(const std::vector<gam::executor_id> &d, //
			gam::private_ptr<T> &&p, //
			PolicyArgs&&... __a) {
		p.push(dist(d, std::forward<PolicyArgs>(__a)...));
	}

	template<typename T>
	void broadcast(const std::vector<gam::executor_id> &d, //
			const gam::public_ptr<T> &p) {
		for (auto to : d) {
			if ( to != gam::rank() )
				p.push(to);
		}
	}

	template<typename T>
	void broadcast(const std::vector<gam::executor_id> &d, //
			gam::private_ptr<T> &&p) {
		USRASSERT(!p.get().is_address());
		for (auto to : d)
			gam::private_ptr<T>(p.get()).push(to);
	}

	Policy dist;
};

} /* namespace gff */

#endif /* FF_D_GFF_DISPATCHER_FAIMILES_BUILTIN_HPP_ */
