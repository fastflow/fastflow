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
 * @file Filter.hpp
 * @author Maurizio Drocco
 * @date Apr 12, 2017
 *
 */

#ifndef FF_D_GFF_FILTER_HPP_
#define FF_D_GFF_FILTER_HPP_

#include <gam.hpp>

#include "Node.hpp"
#include "Logger.hpp"

namespace gff {

/**
 *
 * @brief Input-Output GFF node.
 *
 * \see Node for the general execution model of GFF nodes.
 *
 * Filter represents a node that, during its core processing phase,
 * receives an input pointer, processes it and returns an output pointer.
 *
 * @todo feedback
 */
template<typename InComm, typename OutComm, //
		typename in_t, typename out_t, //
		typename Logic>
class Filter: public Node {

public:
	Filter(InComm &in_comm, OutComm &out_comm) :
			in_comm(in_comm), out_comm(out_comm) {
	}

protected:
#if 0
	/* @brief shares with a downstream node, according to communicator logic
	 * @ingroup GFF-bb-api
	 *
	 * To be called from svc_init() or svc().
	 * It does *not* activate downstream svc().
	 */
	template<typename element_t>
	void share(const gam::public_ptr<element_t> &to_emit)
	{
		Node::put_via(out_comm, to_emit);
	}

	/* @brief broadcast to all downstream nodes
	 * @ingroup GFF-bb-api
	 *
	 * To be called from svc_init().
	 * It does *not* activate downstream svc() or svc().
	 */
	template<typename element_t>
	void broadcast(const gam::public_ptr<element_t> &to_emit)
	{
		Node::broadcast_via(out_comm, to_emit);
	}

	/* @brief get a public pointer shared/broadcasted from upstream node
	 * @ingroup GFF-bb-api
	 *
	 * To be called from svc_init() or svc().
	 */
	template<typename collected_t>
	gam::public_ptr<collected_t> get()
	{
		return Node::get_via<InComm, gam::public_ptr<collected_t>>(in_comm);
	}
#endif

private:
	void set_links() {
		out_comm.internals.source(Node::id__);
		in_comm.internals.destination(Node::id__);
	}

	void run() {
		GFF_PROFILER_TIMER (t0);
		GFF_PROFILER_TIMER (t1);
		GFF_PROFILER_DURATION (d_get);
		GFF_PROFILER_DURATION (d_get_max);
		GFF_PROFILER_COND (unsigned long long d_get_max_index = 0);
		GFF_PROFILER_DURATION (d_svc);

		GFF_LOGLN("FLT start");

		logic.svc_init(out_comm);

		in_t in;
		token_t out;

		bool svc_termination = false;
		while (true) {
			/* prepare local input pointer */
			GFF_PROFILER_HRT(t0);
			in = in_comm.internals.template get<in_t>();
			GFF_PROFILER_HRT(t1);
			GFF_PROFILER_DURATION_ADD(d_get, t0, t1);
			GFF_PROFILER_BOOL(flg, time_diff(t0, t1) > d_get_max);
			GFF_PROFILER_BOOL_COND (flg, d_get_max = time_diff(t0,t1););
			GFF_PROFILER_BOOL_COND (flg, d_get_max_index = token_id;);

			if (!is_eos(in)) { //meaningful input token
				GFF_LOGLN_OS("FLT got=" << in);

				/* process token by invoking user function */
				GFF_PROFILER_HRT(t0);
				out = logic.svc(in, out_comm);
				GFF_PROFILER_HRT(t1);
				GFF_PROFILER_DURATION_ADD(d_svc, t0, t1);

				/* ckeck if meaningful output token */
				if (is_eos(out)) { //user function asked for termination
					GFF_LOGLN_OS("FLT svc returned eos");
					svc_termination = true;
					break;
				}

				else {
					//user function returned go_on: continue to next token
					GFF_LOGLN_OS("FLT svc returned go_on");
					continue;
				}
			}

			else {
				//got eos from upstream communicator
				GFF_LOGLN_OS("FLT got eos");

				if (++received_eos == in_comm.internals.in_cardinality())
					break;
			}

			++token_id;
		}

		assert(!svc_termination || received_eos == 0); //todo error reporting

		logic.svc_end(out_comm);

		/* propagate eos token */
		out_comm.internals.broadcast(global_eos<out_t>());

		/* write profiling */
		GFF_PROFLN("FLT get      = %f s", d_get.count());
		GFF_PROFLN("FLT get MAX  = %f s", d_get_max.count());
		GFF_PROFLN("FLT get MAXi = %d", d_get_max_index);
		GFF_PROFLN("FLT svc      = %f s", d_svc.count());

		GFF_LOGLN("FLT done");
	}

	InComm &in_comm;
	OutComm &out_comm;
	gam::executor_id received_eos = 0;
	Logic logic;
	unsigned long long token_id = 0;
};

} /* namespace gam */

#endif /* FF_D_GFF_FILTER_HPP_ */
