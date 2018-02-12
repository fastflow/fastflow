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

/*
 * option_filter.hpp
 *
 *  Created on: Jun 9, 2017
 *      Author: drocco
 */

#ifndef EXAMPLES_STOCK_MARKET_OPTION_FILTER_HPP_
#define EXAMPLES_STOCK_MARKET_OPTION_FILTER_HPP_

#include <ff/d/gff/gff.hpp>
#include <gam.hpp>

#include "defs.hpp"

template<typename Comm>
class OptionFilterLogic {
public:
    OptionFilterLogic()
            : rdist(0, 1)
    {
    }

    gff::token_t svc(gam::private_ptr<option_batch_t> &in, Comm &c)
    {
        auto lob = in.local();

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

        c.emit(gam::private_ptr<option_batch_t>(std::move(lob)));
        return gff::go_on;
    }

    void svc_init()
    {
    }

    void svc_end()
    {
    }

private:
    std::mt19937 rng;
    std::uniform_real_distribution<double> rdist;
};

#endif /* EXAMPLES_STOCK_MARKET_OPTION_FILTER_HPP_ */
