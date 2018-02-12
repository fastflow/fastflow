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
 * @file        seq_stock_market.cpp
 * @author      Maurizio Drocco
 *
 */

#include <cassert>
#include <random>
#include <sstream>
#include <fstream>

#include <ff/pipeline.hpp>

#include "config.hpp"
#include "black_scholes.hpp"

/*
 * application specific
 */
struct option_t {
    unsigned char name[8];
    OptionData opt;
    double price;
};

struct option_batch_t {
    option_t data[BATCH_SIZE];
    unsigned size = 0;
};

/*
 * farm components
 */
class Emitter: public ff::ff_node {
    void *svc(void *)
    {

        /* buffer some options from the input file */
        std::ifstream in_file(in_fname);
        assert(in_file.is_open());
        option_batch_t obuf;

        for (unsigned i = 0; i < BATCH_SIZE; ++i)
        {
            option_t &o(obuf.data[i]);

            std::string opt_line;
            assert(std::getline(in_file, opt_line, '\n'));
            char otype;
            std::stringstream ins(opt_line);

            /* read stock option data */
            ins >> o.name;
            ins >> o.opt.s >> o.opt.strike >> o.opt.r >> o.opt.divq;
            ins >> o.opt.v >> o.opt.t >> otype >> o.opt.divs;
            ins >> o.opt.DGrefval;
            o.opt.OptionType = (otype == 'P');
        }

        /* create a stream of random options */
        std::mt19937 rng;
        std::uniform_int_distribution<> odist(0, BATCH_SIZE - 1);
        for (unsigned long long bi = 0; bi < STREAM_LEN / BATCH_SIZE; ++bi)
        {
            /* create a batch */
            auto lob = new option_batch_t();

            /* fill the batch randomly from the buffer */
            for (lob->size = 0; lob->size < BATCH_SIZE; ++lob->size)
                lob->data[lob->size] = obuf.data[odist(rng)];

            /* emit the batch */
            ff_send_out(lob);
        }

        in_file.close();

        return EOS;
    }
};

class Worker: public ff::ff_node {
public:
    Worker()
            : rdist(0, 1)
    {
    }

    void *svc(void *t)
    {
        auto lob = (option_batch_t *) t;

        for (unsigned bi = 0; bi < lob->size; ++bi)
        {
            /* compute the price with Black&Scholes algorithm */
            double res = black_scholes(lob->data[bi].opt);

            /* extra work for augmenting the grain */
            for (unsigned ri = 0; ri < REBUTTAL; ++ri)
            {
                double res2 = black_scholes(lob->data[bi].opt);
                res = (res * rdist(rng) + res2 * rdist(rng)) / 2;
            }

            lob->data[bi].price = res;
        }

        return lob;
    }

private:
    std::mt19937 rng;
    std::uniform_real_distribution<double> rdist;
};

class Collector: public ff::ff_node {
    void *svc(void *t)
    {
        auto lob = (option_batch_t *) t;

        for (unsigned bi = 0; bi < lob->size; ++bi)
            printf("<%s, %f>\n", lob->data[bi].name, lob->data[bi].price);

        return GO_ON;
    }
};

/*
 *******************************************************************************
 *
 * main
 *
 *******************************************************************************
 */
int main(int argc, char * argv[])
{
    Emitter e;
    Worker w;
    Collector c;

    ff::ff_pipeline pipe;
    pipe.add_stage(&e);
    pipe.add_stage(&w);
    pipe.add_stage(&c);

    pipe.run_and_wait_end();

    /* print parameters and measure in a single line */
    printf("len\ttime\n");
    printf("%d\t%f\n", STREAM_LEN, pipe.ffTime() / 1000);

    return 0;
}
