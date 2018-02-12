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
 * @file 2Sink.hpp
 * @author Maurizio Drocco
 * @date Apr 12, 2017
 *
 */

#ifndef FF_D_GFF_SINK_HPP_
#define FF_D_GFF_SINK_HPP_

#include <gam.hpp>

#include "Node.hpp"
#include "Logger.hpp"

namespace gff {

/**
 *
 * @brief Input-only GFF node.
 *
 * \see Node for the general execution model of GFF nodes.
 *
 * Sink represents a GFF node which, during its core processing phase,
 * receives an input token, processes it and performs some internal action
 * without returning anything.
 *
 * @todo feedback
 */
template<typename InComm, typename in_t, typename Logic>
class Sink: public Node {
public:
	Sink(InComm &comm) :
			in_comm(comm) {
	}

	virtual ~Sink() {
	}

protected:
#if 0
	/* @brief \see Filter::get()
	 * @ingroup GFF-bb-api
	 */
	template<typename collected_t>
	gam::public_ptr<collected_t> get()
	{
		return Node::get_<InComm, collected_t>(in_comm);
	}
#endif

private:
	void set_links() {
		in_comm.internals.destination(Node::id__);
	}

	void run() {
		GFF_PROFILER_TIMER (t0);
		GFF_PROFILER_TIMER (t1);
		GFF_PROFILER_DURATION (d_get);
		GFF_PROFILER_DURATION (d_get_max);
		GFF_PROFILER_COND (unsigned long long d_get_max_index = 0);
		GFF_PROFILER_DURATION (d_svc);

		GFF_LOGLN("SNK start");

		Node::init_(logic);

		in_t in;

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
				GFF_LOGLN_OS("SNK got=" << in);

				/* process token by invoking user function */
				GFF_PROFILER_HRT(t0);
				logic.svc(in);
				GFF_PROFILER_HRT(t1);
				GFF_PROFILER_DURATION_ADD(d_svc, t0, t1);
			}

			else {
				//got eos from upstream communicator
				GFF_LOGLN_OS("SNK got eos");

				if (++received_eos == in_comm.internals.in_cardinality())
					break;
			}

			++token_id;
		}

		Node::end_(logic);

		/* write profiling */
		GFF_PROFLN("SNK svc      = %f s", d_svc.count());
		GFF_PROFLN("SNK get      = %f s", d_get.count());
		GFF_PROFLN("SNK get MAX  = %f s", d_get_max.count());
		GFF_PROFLN("SNK get MAXi = %d", d_get_max_index);

		/* write single-line profiling */
		GFF_PROFLN_RAW("svc\tget\tgetM\tgetMi");
		GFF_PROFLN_RAW("%f\t%f\t%f\t%d", //
				d_svc.count(), d_get.count(), //
				d_get_max.count(), d_get_max_index);

		GFF_LOGLN("SNK done");
	}

	InComm &in_comm;
	gam::executor_id received_eos = 0;
	Logic logic;
	unsigned long long token_id = 0;
};

} /* namespace gff */

#endif /* FF_D_GFF_SINK_HPP_ */
