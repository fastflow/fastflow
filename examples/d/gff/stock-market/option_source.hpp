/*
 * option_source.hpp
 *
 *  Created on: Jun 9, 2017
 *      Author: drocco
 */

#ifndef EXAMPLES_STOCK_MARKET_OPTION_SOURCE_HPP_
#define EXAMPLES_STOCK_MARKET_OPTION_SOURCE_HPP_

#include <gam.hpp>
//#include <ff/d/gff/utils.hpp>

#include "defs.hpp"

/*
 * Emulates reading a stock options stream
 */
template<typename Comm>
class OptionSourceLogic {
public:

    gff::token_t svc(Comm &c)
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
            auto obatch = gam::make_private<option_batch_t>();
            auto lob = obatch.local();

            /* fill the batch randomly from the buffer */
            for (lob->size = 0; lob->size < BATCH_SIZE; ++lob->size)
                lob->data[lob->size] = obuf.data[odist(rng)];

            /* emit the batch */
            c.emit(gam::private_ptr<option_batch_t>(std::move(lob)));
        }

        in_file.close();

        return gff::eos;
    }

    void svc_init()
    {
    }

    void svc_end()
    {
    }

private:
    unsigned emitted_frames = 0;
};

#endif /* EXAMPLES_STOCK_MARKET_OPTION_SOURCE_HPP_ */
