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
 * @file        gff_bb_farm.cpp
 * @brief       a simple gff farm with building blocks
 * @author      Maurizio Drocco
 * 
 * This example is a simple farm, in which each component represents
 * a relevant instance of GFF node:
 * - Emitter is a SOURCE node:
 *   it produces a stream (no input channel)
 * - Worker is a (type-changing) FILTER:
 *   it reads an input stream from its input channel and emits an output stream
 *   of different type on its output channel
 * - Collector is a SINK node:
 *   it merges and consumes the streams coming from Workers (no output channel)
 *
 * In this examples, output items from workers are casted to char in order to
 * neutralize any effect of the processing order in the summing collector.
 *
 */

#include <iostream>
#include <cassert>
#include <cmath>

#include <ff/d/gff/gff.hpp>

#define NWORKERS                 2
#define STREAMLEN             32
#define NUMBER             4

/*
 * To define a gff node, the user has to define first an internal class (a.k.a.
 * dff logic), with the following functions:
 * - svc_init: called once at the beginning of node's execution
 * - svc_end:  called once at the end of node's execution
 * - svc:      called repeatedly during node's execution
 *
 * The signature of svc varies depending on node's family (source, processor,
 * sink). See sample codes.
 *
 * To complete the definition of a gff node, the internal class is passed as
 * template parameter to the generic class corresponding to the node's family.
 */

/*
 *******************************************************************************
 *
 * farm components
 *
 *******************************************************************************
 */
/*
 * gff logic generating a stream of random integers
 */
class EmitterLogic {
public:
	/**
	 * The svc function is called repeatedly by the runtime, until an eos
	 * token is returned.
	 * Pointers are sent downstream by calling the emit() function on the
	 * output channel, that is passed as input argument.
	 *
	 * @param c is the output channel (could be a template for simplicity)
	 * @return a gff token
	 */
	gff::token_t svc(gff::OutBundleBroadcast<gff::OneToOne> &c) {
		if (n++ < STREAMLEN) {
			c.emit(gam::make_public<int>(NUMBER));
			return gff::go_on;
		}
		return gff::eos;
	}

	void svc_init() {
	}

	void svc_end(gff::OutBundleBroadcast<gff::OneToOne> &c) {
	}

private:
	unsigned n = 0;
};

/*
 * define a Source node with the following template parameters:
 * - the type of the output channel
 * - the type of the emitted pointers
 * - the gff logic
 */
typedef gff::Source<gff::OutBundleBroadcast<gff::OneToOne>, //
		gam::public_ptr<int>, //
		EmitterLogic> Emitter;

/*
 * gff logic low-passing input integers below THRESHOLD and computes sqrt
 */
class WorkerLogic {
public:
	/**
	 * The svc function is called upon each incoming pointer from upstream.
	 * Pointers are sent downstream by calling the emit() function on the
	 * output channel, that is passed as input argument.
	 *
	 * @param in is the input pointer
	 * @param c is the output channel (could be a template for simplicity)
	 * @return a gff token
	 */
	gff::token_t svc(gam::public_ptr<int> &in, gff::NondeterminateMerge &c) {
		auto local_in = in.local();
		c.emit(gam::make_private<char>((char) *local_in));
		return gff::go_on;
	}

	void svc_init(gff::NondeterminateMerge &c) {
	}

	void svc_end(gff::NondeterminateMerge &c) {
	}
};

/*
 * define a Source node with the following template parameters:
 * - the type of the input channel
 * - the type of the output channel
 * - the type of the input pointers
 * - the type of the output pointers
 * - the gff logic
 */
typedef gff::Filter<gff::OneToOne, gff::NondeterminateMerge, //
		gam::public_ptr<int>, gam::private_ptr<char>, //
		WorkerLogic> Worker;

/*
 * gff logic summing up all filtered tokens and finally checking the result
 */
class CollectorLogic {
public:
	/**
	 * The svc function is called upon each incoming pointer from upstream.
	 *
	 * @param in is the input pointer
	 * @return a gff token
	 */
	void svc(gam::private_ptr<char> &in) {
		auto local_in = in.local();
		std::cout << (int) *local_in << std::endl;
		assert(*local_in == NUMBER);
	}

	void svc_init() {
	}

	/*
	 * at the end of processing, check the result
	 */
	void svc_end() {
	}

private:
	int sum = 0;
};

/*
 * define a Source node with the following template parameters:
 * - the type of the input channel
 * - the type of the input pointers
 * - the gff logic
 */
typedef gff::Sink<gff::NondeterminateMerge, //
		gam::private_ptr<char>, //
		CollectorLogic> Collector;

/*
 *******************************************************************************
 *
 * main
 *
 *******************************************************************************
 */
int main(int argc, char * argv[]) {
	/*
	 * Create the channels for inter-node communication.
	 * A channel can carry both public and private pointers.
	 */
	gff::NondeterminateMerge w2c;
	gff::OutBundleBroadcast<gff::OneToOne> e2w;
	e2w.internals.add_comm();
	e2w.internals.add_comm();

	/*
	 * In this preliminary implementation, a single global network is
	 * created and nodes can be added only to the global network.
	 */

	/* bind nodes to channels and add to the network */
	gff::add(Emitter(e2w)); //e2w is the emitter's output channel
	for (unsigned i = 0; i < NWORKERS; ++i) {
		gff::add(Worker(*e2w.get(i%2), w2c)); //e2w/w2c are the workers' i/o channels
	}
	gff::add(Collector(w2c)); //w2c is the collector's input channel

	/* execute the network */
	gff::run();

	return 0;
}
