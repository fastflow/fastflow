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
 * @file        Node.hpp
 * @author      Maurizio Drocco
 * 
 */
#ifndef FF_D_GFF_NODE_HPP_
#define FF_D_GFF_NODE_HPP_

#include "defs.hpp"

namespace gff {

/*
 *
 */
class Node {
public:
	virtual ~Node() {

	}

	void id(gam::executor_id i) {
		id__ = i;
		set_links();
	}

	gam::executor_id id() {
		return id__;
	}

	virtual void set_links() = 0;

	virtual void run() = 0;

protected:
	template<typename Logic>
	void init_(Logic &l) {
		l.svc_init();
	}

	template<typename Logic>
	void end_(Logic &l) {
		l.svc_end();
	}

protected:
	gam::executor_id id__;
};

} /* namespace gff */

#endif /* FF_D_GFF_NODE_HPP_ */
