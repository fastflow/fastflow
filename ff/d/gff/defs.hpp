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
 * @file defs.hpp
 * @author Maurizio Drocco
 * @date Apr 14, 2017
 */

#ifndef FF_D_GFF_DEFS_HPP_
#define FF_D_GFF_DEFS_HPP_

#include <gam.hpp>

namespace gff {
/*
 ***************************************************************************
 *
 * synchronization token
 *
 ***************************************************************************
 */
typedef uint64_t token_t;
static constexpr token_t eos = gam::GlobalPointer::last_reserved;
static constexpr uint64_t go_on = eos - 1;

/*
 ***************************************************************************
 *
 * end-of-stream token
 *
 ***************************************************************************
 */
template<typename T>
static inline bool is_eos(const gam::public_ptr<T> &token) {
	return token.get().address() == eos;
}

template<typename T>
static inline bool is_eos(const gam::private_ptr<T> &token) {
	return token.get().address() == eos;
}

static inline bool is_eos(const token_t &token) {
	return token == eos;
}

/*
 * builds a global pointer representing an eos token.
 */
template<typename T>
T global_eos() {
	return T(gam::GlobalPointer(eos));
}

/*
 ***************************************************************************
 *
 * go-on token
 *
 ***************************************************************************
 */
static inline bool is_go_on(const token_t &token) {
	return token == go_on;
}

} /* namespace gff */

#endif /* FF_D_GFF_DEFS_HPP_ */
