/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file bitflags.hpp
 *  \ingroup building_blocks
 *
 *
 */

/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */
#ifndef FF_BITFLAGS_HPP
#define FF_BITFLAGS_HPP

#include <vector>
#include <tuple>

namespace ff {

/**
 * Flags used in the  \ref setInPtr and \ref setOutPtr methods for 
 * providing commands to the run-time concerning H2D and D2H data transfers 
 * and device memory allocation.
 *
 */
enum class BitFlags {
        COPYTO,              
        DONTCOPYTO,          
        REUSE, 
        DONTREUSE, 
        RELEASE, 
        DONTRELEASE, 
        COPYBACK,
        DONTCOPYBACK
};

using bitflagsVector = std::vector<std::tuple<BitFlags,BitFlags,BitFlags> > ;

// TO BE IMPLEMENTED: the current version is just a test case
static inline const bitflagsVector extractFlags(const std::string &str, const int kernel_id) {
    bitflagsVector V(10);
    for(size_t i=0;i<V.size();++i)
        V[i] = std::make_tuple<BitFlags,BitFlags,BitFlags>(BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);
    return V;
}
    
static inline BitFlags getCopy(int pos, const bitflagsVector &V) {
    return std::get<0>(V[pos]);
}
static inline BitFlags getReuse(int pos, const bitflagsVector &V) {
    return std::get<1>(V[pos]);
}
static inline BitFlags getRelease(int pos, const bitflagsVector &V) {
    return std::get<2>(V[pos]);
}

// *******************************************************************


} // namespace


#endif /* FF_BITFLAGS_HPP */
