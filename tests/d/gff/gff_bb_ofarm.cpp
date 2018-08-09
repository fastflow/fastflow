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
 * @file        gff_bb_ofarm.cpp
 * @brief       a simple gff ordering farm with building blocks
 * @author      Maurizio Drocco
 * 
 * This example is a simple ordering farm.
 * (\see gff_bb_farm.cpp)
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
class EmitterLogic
{
private:
    unsigned n = 0;
    std::mt19937 rng;

public:
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

    void svc_end(gff::RoundRobinSwitch &c)
    {
    }
};

typedef gff::Source<gff::RoundRobinSwitch, gam::private_ptr<int>, //
        EmitterLogic> Emitter;

/*
 * Lowpass selects input integers below THRESHOLD
 */
class WorkerLogic
{
public:
    gff::token_t svc(gam::private_ptr<int> &in, gff::RoundRobinMerge &c)
    {
        auto local_in = in.local();
        c.emit(gam::make_private<float>(std::sqrt(*local_in)));
        return gff::go_on;
    }

    void svc_init(gff::RoundRobinMerge &c)
    {
    }

    void svc_end(gff::RoundRobinMerge &c)
    {
    }
};

typedef gff::Filter<gff::RoundRobinSwitch, gff::RoundRobinMerge, //
        gam::private_ptr<int>, gam::private_ptr<float>, //
        WorkerLogic> Worker;

/*
 * Sink sums up and check
 */
class CollectorLogic
{
private:
    float sum = 0;
    std::mt19937 rng;

public:
    void svc(gam::private_ptr<float> &in)
    {
        auto local_in = in.local();
        std::cout << *local_in << std::endl;
        sum += *local_in;
    }

    void svc_init()
    {
    }

    void svc_end()
    {
        float res = 0;
        for (unsigned i = 0; i < STREAMLEN; ++i)
        {
            int x = (int) (rng() % RNG_LIMIT);
            res += std::sqrt(x);
        }
        if (res != sum)
        {
            fprintf(stderr, "sum=%f exp=%f\n", sum, res);
            exit(1);
        }
    }
};

typedef gff::Sink<gff::RoundRobinMerge, gam::private_ptr<float>, //
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
    gff::RoundRobinMerge w2c;

    gff::add(Emitter(e2w));
    for (unsigned i = 0; i < NWORKERS; ++i)
        gff::add(Worker(e2w, w2c));
    gff::add(Collector(w2c));

    gff::run();

    return 0;
}
