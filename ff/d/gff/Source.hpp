/* ***************************************************************************
 *
 *  This file is part of FastFlow.
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  FastFlow is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with FastFlow. If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************
 */

/**
 * @file Source.hpp
 * @author Maurizio Drocco
 * @date Apr 12, 2017
 *
 */

#ifndef FF_D_GFF_SOURCE_HPP_
#define FF_D_GFF_SOURCE_HPP_

#include <gam.hpp>

#include "Node.hpp"
#include "Logger.hpp"

namespace gff {

/**
 *
 * @brief Output-only GFF node.
 *
 * \see GFFNode for the general execution model of GFF nodes.
 *
 * GFFFilter represents a GFF node which, during its core processing phase,
 * performs some internal action and returns a brand new token.
 *
 * @todo feedback
 */
template<typename OutComm, typename out_t, typename Logic>
class Source: public Node {
public:
	Source(OutComm &comm) :
			out_comm(comm) {
	}

	virtual ~Source() {
	}

protected:

#if 0
	/* @brief \see Filter::share()
	 * @ingroup GFF-bb-api
	 *
	 * It does *not* activate downstream svc().
	 */
	void share(const gam::public_ptr<out_t__> &to_emit)
	{
		Node::share_(out_comm, to_emit);
	}

	/* @brief \see GFFFilter::broadcast()
	 * @ingroup GFF-bb-api
	 *
	 * It does *not* activate downstream svc().
	 */
	template<typename T>
	void broadcast(const gam::public_ptr<T> &to_emit)
	{
		GFFNode::broadcast_(out_comm, to_emit);
	}
#endif

private:
	void set_links() {
		out_comm.internals.source(Node::id__);
	}

	void run() {
		GFF_PROFILER_TIMER (t0);
		GFF_PROFILER_TIMER (t1);
		GFF_PROFILER_DURATION (d_svc);

		GFF_LOGLN("SRC start");

		Node::init_(logic);
		token_t out;

		while (true) {
			/* generate a token by invoking user function */
			GFF_PROFILER_HRT(t0);
			out = logic.svc(out_comm);
			GFF_PROFILER_HRT(t1);
			GFF_PROFILER_DURATION_ADD(d_svc, t0, t1);

			/* ckeck if meaningful output token */
			if (is_eos(out)) { //user function asked for termination
				GFF_LOGLN_OS("SRC svc returned eos");
				break;
			}

			else {
				//user function returned go_on: skip to next token
				GFF_LOGLN_OS("SRC svc returned go_on");
				continue;
			}
		}

		logic.svc_end(out_comm);

		/* broadcast eos token */
		out_comm.internals.broadcast(global_eos<out_t>());

		/* write profiling */
		GFF_PROFLN("SRC svc  = %f s", d_svc.count());

		GFF_LOGLN("SRC done");
	}

	OutComm &out_comm;
	Logic logic;
};

} /* namespace gff */

#endif /* FF_D_GFF_SOURCE_HPP_ */
