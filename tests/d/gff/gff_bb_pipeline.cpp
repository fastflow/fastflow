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
 * @file        gff_bb_pipeline.cpp
 * @brief       a simple gff pipeline with building blocks
 * @author      Maurizio Drocco
 * 
 * This example is a simple 4-stage pipeline:
 * - stage 1 generates a stream of random integers within a range
 * - stage 2 is a low-pass filter that discards all numbers above a threshold
 * - stage 3 computes sqrt
 * - stage 4 writes the stream to standard output
 *
 * Each stage represents a relevant instance of GFF node:
 * - Stage 1 is a SOURCE node:
 *   it produces a stream (no input channel)
 * - Stage 2 is a type-preserving FILTER:
 *   it reads an input stream from its input channel and emits an output stream
 *   of the same type on its output channel
 * - Stage 3 is a type-changing FILTER
 * - Stage 4 is a SINK node:
 * it consumes a stream (no output channel)
 *
 */

#include <iostream>
#include <cassert>
#include <cmath>

#include <ff/d/gff/gff.hpp>
#include <TrackingAllocator.hpp>

#define STREAMLEN             1024
#define RNG_LIMIT             1000
#define THRESHOLD  (RNG_LIMIT / 2)

/*
 * we use unicast communicators to implement 1-to-1 pipeline channels
 */

/*
 ***************************************************************************
 *
 * pipeline stages
 *
 ***************************************************************************
 */
/*
 * Source generates a stream of random integers within [0, RNG_LIMIT)
 */
class PipeSourceLogic {
private:
    unsigned n = 0;
    std::mt19937 rng;

public:
    gff::token_t svc(gff::OneToOne &c)
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

    void svc_end(gff::OneToOne &c)
    {
    }
};

typedef gff::Source<gff::OneToOne, //
        gam::private_ptr<int>, //
        PipeSourceLogic> PipeSource;

/*
 * Lowpass selects input integers below THRESHOLD
 */
class LowpassLogic {
public:
    gff::token_t svc(gam::private_ptr<int> &in, gff::OneToOne &c)
    {
        auto local_in = in.local();
        if (*local_in < THRESHOLD)
            c.emit(gam::private_ptr<int>(std::move(local_in)));
        return gff::go_on;
    }

    void svc_init(gff::OneToOne &c)
    {
    }

    void svc_end(gff::OneToOne &c)
    {
    }
};

typedef gff::Filter<gff::OneToOne, gff::OneToOne, //
        gam::private_ptr<int>, gam::private_ptr<int>, //
        LowpassLogic> Lowpass;

/*
 * Sqrt computes sqrt of each input integer and emits the results as floats
 */
class SqrtLogic {
private:
    float sum = 0;
    std::mt19937 rng;

public:
    gff::token_t svc(gam::private_ptr<int> &in, gff::OneToOne &c)
    {
        float res = std::sqrt(*(in.local()));
        sum += res;
        c.emit(gam::make_private<float>(res));
        return gff::go_on;
    }

    void svc_init(gff::OneToOne &c)
    {
    }

    void svc_end(gff::OneToOne &c)
    {
        float res = 0;
        for (unsigned i = 0; i < STREAMLEN; ++i)
        {
            int x = (int) (rng() % RNG_LIMIT);
            if (x < THRESHOLD)
                res += sqrt(x);
        }
        assert(res == sum);
    }
};

typedef gff::Filter<gff::OneToOne, gff::OneToOne, //
        gam::private_ptr<int>, gam::private_ptr<float>, //
        SqrtLogic> Sqrt;

/*
 * Sink prints its input stream
 */
class PipeSinkLogic {
public:
    void svc(gam::private_ptr<float> &in)
    {
        std::cout << *(in.local()) << std::endl;
    }

    void svc_init()
    {
    }

    void svc_end()
    {
    }
};

typedef gff::Sink<gff::OneToOne, //
        gam::private_ptr<float>, //
        PipeSinkLogic> PipeSink;

/*
 *******************************************************************************
 *
 * main
 *
 *******************************************************************************
 */
int main(int argc, char * argv[])
{
    gff::OneToOne comm1, comm2, comm3;

    gff::add(PipeSource(comm1));
    gff::add(Lowpass(comm1, comm2));
    gff::add(Sqrt(comm2, comm3));
    gff::add(PipeSink(comm3));

    gff::run();

    return 0;
}
