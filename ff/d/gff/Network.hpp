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
 * @file        Network.hpp
 * @brief       implements Network class.
 * @author      Maurizio Drocco
 * 
 */
#ifndef FF_D_GFF_NETWORK_HPP_
#define FF_D_GFF_NETWORK_HPP_

#include <cassert>

#include <gam/gam.hpp>

#include "Node.hpp"
#include "Logger.hpp"
#include "Profiler.hpp"

namespace gff {

/*
 * Singleton class representing a whole GFF application.
 */
class Network {
public:
	static Network *getNetwork() {
		static Network network;
		return &network;
	}

	~Network() {
		std::vector<Node *>::size_type i;
		for (i = 0; i < gam::rank() && i < cardinality(); ++i)
			delete nodes[i];
		if (i < cardinality())
			assert(nodes[i] == nullptr);
		for (++i; i < cardinality(); ++i)
			delete nodes[i];
		nodes.clear();
	}

	gam::executor_id cardinality() {
		return nodes.size();
	}

	template<typename T>
	void add(const T &n) {
		auto np = dynamic_cast<Node *>(new T(n));
		np->id((gam::executor_id) nodes.size());
		nodes.push_back(np);
	}

	template<typename T>
	void add(T &&n) {
		auto np = dynamic_cast<Node *>(new T(std::move(n)));
		np->id((gam::executor_id) nodes.size());
		nodes.push_back(np);
	}

	void run() {
		GFF_PROFILER_TIMER(t0);
		GFF_PROFILER_TIMER(t_init);
		GFF_PROFILER_TIMER(t_run);

		GFF_PROFILER_HRT(t0);

		/* initialize the logger */
		char *env = std::getenv("GAM_LOG_PREFIX");
		assert(env);
		GFF_LOGGER_INIT(env, gam::rank());

		/* initialize the profiler */
		GFF_PROFILER_INIT(env, gam::rank());

		GFF_PROFILER_HRT(t_init);

		/* check cardinality */
		GFF_LOGLN_OS("gam cardinality = " << gam::cardinality());
		GFF_LOGLN_OS("network cardinality = " << cardinality());
		assert(gam::cardinality() >= cardinality()); //todo error reporting

		if (gam::rank() < cardinality()) {
			/* run the node associated to the executor */
			nodes[gam::rank()]->run();

			GFF_PROFILER_HRT(t_run);

			/* call node destructor to trigger destruction of data members */
			delete nodes[gam::rank()];
			nodes[gam::rank()] = nullptr;

		}

		/* write profiling */
		GFF_PROFLN("NET init  = %f s", GFF_PROFILER_SPAN(t0, t_init));
		GFF_PROFLN("NET run   = %f s", GFF_PROFILER_SPAN(t_init, t_run));

		/* write single-line profiling */
		GFF_PROFLN_RAW("init\tsvc");
		GFF_PROFLN_RAW("%f\t%f", //
				GFF_PROFILER_SPAN(t0, t_init),//
				GFF_PROFILER_SPAN(t_init, t_run));

		/* finalize the profiler */
		GFF_PROFILER_FINALIZE(gam::rank());

		/* finalize the logger */
		GFF_LOGGER_FINALIZE(gam::rank());
	}

private:
	std::vector<Node *> nodes;
};

template<typename T>
static void add(const T &n) {
	Network::getNetwork()->add(n);
}

template<typename T>
static void add(T &&n) {
	Network::getNetwork()->add(std::move(n));
}

static void run() {
	Network::getNetwork()->run();
}

} /* namespace gff */

#endif /* FF_D_GFF_NETWORK_HPP_ */
