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
 * @file        builtin.hpp
 * @author      Maurizio Drocco
 * 
 */
#ifndef FF_D_GFF_DISPATCHING_POLICIES_BUILTIN_HPP_
#define FF_D_GFF_DISPATCHING_POLICIES_BUILTIN_HPP_

namespace gff {

#include <gam/gam.hpp>

/*
 *******************************************************************************
 *
 * Pushing Policies
 *
 *******************************************************************************
 */

/*
 *
 */
class ConstantTo {
public:
	gam::executor_id operator()(const std::vector<gam::executor_id> &dest) {
		return dest[0];
	}
};

/*
 *
 */
class RRTo {
public:
	gam::executor_id operator()(const std::vector<gam::executor_id> &dest) {
		gam::executor_id res = dest[rr_cnt];
		rr_cnt = (rr_cnt + 1) % dest.size();
		return res;
	}

private:
	gam::executor_id rr_cnt = 0;
};

/*
 *
 */
class KeyedTo {
public:
	template<typename K>
	gam::executor_id operator()(const std::vector<gam::executor_id> &dest, //
			const K &key) {
		return key % dest.size();
	}
};

/*
 *
 */
class ConstantToAll {
public:
	std::vector< gam::executor_id > operator()(const std::vector<gam::executor_id> &dest) {
		std::vector< gam::executor_id > newdest;
		for (auto to : dest){
			if (to != gam::rank())
				newdest.push_back(to);
		}
		return newdest;
	}
};

/*
 *******************************************************************************
 *
 * Pulling Policies
 *
 *******************************************************************************
 */
/**
 * Constant collection policy
 */
class ConstantFrom {
public:
	gam::executor_id operator()(const std::vector<gam::executor_id> &src) {
		return src[0];
	}
};

/**
 * Round-Robin collection policy
 */
class RRFrom {
public:
	gam::executor_id operator()(const std::vector<gam::executor_id> &src) {
		gam::executor_id res = src[rr_cnt];
		rr_cnt = (rr_cnt + 1) % src.size();
		return res;
	}

private:
	gam::executor_id rr_cnt = 0;
};

} //namespace gff

#endif /* FF_D_GFF_DISPATCHING_POLICIES_BUILTIN_HPP_ */
