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
#include <TrackingAllocator.hpp>

#define NWORKERS                 4
#define STREAMLEN             1024
#define RNG_LIMIT             1000
#define THRESHOLD  (RNG_LIMIT / 2)

/*
 ***************************************************************************
 *
 * farm components
 *
 ***************************************************************************
 */
/*
 * Source generates a stream of random integers within [0, RNG_LIMIT)
 */
class EmitterLogic {
public:
    unsigned n = 0;
    std::mt19937 rng;

    gff::token_t svc(gff::RoundRobinSwitch &c)
    {
        if (n++ < STREAMLEN)
        {
            c.emit(gam::make_private<int>((int) (rng() % RNG_LIMIT)));
            return gff::go_on;
        }
        return gff::eos;
    }

    void svc_init()
    {
    }

    void svc_end()
    {
    }
};

typedef gff::Source<gff::RoundRobinSwitch, //
        gam::private_ptr<int>, //
        EmitterLogic> Emitter;

/*
 * Lowpass selects input integers below THRESHOLD and computes sqrt
 */
class WorkerLogic {
public:
    gff::token_t svc(gam::private_ptr<int> &in, gff::NondeterminateMerge &c)
    {
        auto local_in = in.local();
        if (*local_in < THRESHOLD)
            c.emit(gam::make_private<char>((char) std::sqrt(*local_in)));
        return gff::go_on;
    }

    void svc_init()
    {
    }

    void svc_end()
    {
    }
};

typedef gff::Filter<gff::RoundRobinSwitch, gff::NondeterminateMerge, //
        gam::private_ptr<int>, gam::private_ptr<char>, //
        WorkerLogic> Worker;

/*
 * Collector sums up all filtered tokens
 */
class CollectorLogic {
private:
    int sum = 0;
    std::mt19937 rng;

public:
    void svc(gam::private_ptr<char> &in)
    {
        auto local_in = in.local();
        std::cout << (int) *local_in << std::endl;
        sum += *local_in;
    }

    void svc_init()
    {
    }

    void svc_end()
    {
        int res = 0;
        for (unsigned i = 0; i < STREAMLEN; ++i)
        {
            int x = (int) (rng() % RNG_LIMIT);
            if (x < THRESHOLD)
                res += (char) std::sqrt(x);
        }
        if (res != sum)
        {
            fprintf(stderr, "sum=%d exp=%d\n", sum, res);
            exit(1);
        }
    }
};

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
int main(int argc, char * argv[])
{
    gff::RoundRobinSwitch e2w;
    gff::NondeterminateMerge w2c;

    gff::add(Emitter(e2w));
    for (unsigned i = 0; i < NWORKERS; ++i)
        gff::add(Worker(e2w, w2c));
    gff::add(Collector(w2c));

    gff::run();

    return 0;
}
