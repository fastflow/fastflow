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
 * price_writer.hpp
 *
 *  Created on: Jun 9, 2017
 *      Author: drocco
 */

#ifndef EXAMPLES_STOCK_MARKET_PRICE_WRITER_HPP_
#define EXAMPLES_STOCK_MARKET_PRICE_WRITER_HPP_

#include <gam.hpp>
#include <ff/d/gff/gff.hpp>

#include "defs.hpp"

class PriceWriterLogic {
public:
    gff::token_t svc(gam::private_ptr<option_batch_t> &in)
    {
        auto lob = in.local();

        for (unsigned bi = 0; bi < lob->size; ++bi)
            printf("<%s, %f>\n", lob->data[bi].name, lob->data[bi].price);

        return gff::go_on;
    }

    void svc_init()
    {
    }

    void svc_end()
    {
    }

};

#endif /* EXAMPLES_STOCK_MARKET_PRICE_WRITER_HPP_ */
