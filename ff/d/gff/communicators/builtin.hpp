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
#ifndef FF_D_GFF_COMMUNICATORS_BUILTIN_HPP_
#define FF_D_GFF_COMMUNICATORS_BUILTIN_HPP_

#include <gam.hpp>

#include "../dispatcher_faimiles/builtin.hpp"
#include "../dispatching_policies/builtin.hpp"
#include "../CommunicatorInternals.hpp"

namespace gff {
template<typename T, typename Internals, typename ... PolicyArgs>
static inline void emit_(const gam::public_ptr<T> &p, Internals &internals, //
		PolicyArgs&&... __a) {
	internals.put(p, std::forward<PolicyArgs>(__a)...);
}

template<typename T, typename Internals, typename ... PolicyArgs>
static inline void emit_(gam::private_ptr<T> &&p, Internals &internals, //
		PolicyArgs&&... __a) {
	internals.put(std::move(p), std::forward<PolicyArgs>(__a)...);
}

class OneToOne {
public:
	template<typename T>
	void emit(const gam::public_ptr<T> &p) {
		emit_(p, internals);
	}

	template<typename T>
	void emit(gam::private_ptr<T> &&p) {
		emit_(std::move(p), internals);
	}

	CommunicatorInternals<Switch<ConstantTo>, Merge<ConstantFrom>> internals;
};

class OneToAll {
public:
	template<typename T>
	void emit(const gam::public_ptr<T> &p) {
		emit_(p, internals);
	}

	CommunicatorInternals<Multicast<ConstantToAll>, Merge<ConstantFrom>> internals;
};

class NDOneToAll {
public:
	template<typename T>
	void emit(const gam::public_ptr<T> &p) {
		emit_(p, internals);
	}

	CommunicatorInternals<Multicast<ConstantToAll>, NDMerge> internals;
};

class RoundRobinSwitch {
public:
	template<typename T>
	void emit(const gam::public_ptr<T> &p) {
		emit_(p, internals);
	}

	template<typename T>
	void emit(gam::private_ptr<T> &&p) {
		emit_(std::move(p), internals);
	}

	CommunicatorInternals<Switch<RRTo>, Merge<ConstantFrom>> internals;
};

class Shuffle {
public:
	template<typename T>
	void emit(const gam::public_ptr<T> &p) {
		emit_(p, internals);
	}

	template<typename T>
	void emit(gam::private_ptr<T> &&p) {
		emit_(std::move(p), internals);
	}

	CommunicatorInternals<Switch<KeyedTo>, Merge<ConstantFrom>> internals;
};

class RoundRobinMerge {
public:
	template<typename T>
	void emit(const gam::public_ptr<T> &p) {
		emit_(p, internals);
	}

	template<typename T>
	void emit(gam::private_ptr<T> &&p) {
		emit_(std::move(p), internals);
	}

	CommunicatorInternals<Switch<ConstantTo>, Merge<RRFrom>> internals;
};

class NondeterminateMerge {
public:
	template<typename T>
	void emit(const gam::public_ptr<T> &p) {
		emit_(p, internals);
	}

	template<typename T>
	void emit(gam::private_ptr<T> &&p) {
		emit_(std::move(p), internals);
	}

	CommunicatorInternals<Switch<ConstantTo>, NDMerge> internals;
};

} /* namespace gff */

#endif /* FF_D_GFF_COMMUNICATORS_BUILTIN_HPP_ */
