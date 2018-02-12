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
 * defs.hpp
 *
 *  Created on: Jun 9, 2017
 *      Author: drocco
 */

#ifndef EXAMPLES_STOCK_MARKET_BATCH_DEFS_HPP_
#define EXAMPLES_STOCK_MARKET_BATCH_DEFS_HPP_

#include <ff/d/gff/gff.hpp>

#include "config.hpp"
#include "black_scholes.hpp"

/*
 * communicators
 */
gff::RoundRobinSwitch e2w;
gff::RoundRobinMerge w2c;

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

#endif /* EXAMPLES_STOCK_MARKET_BATCH_DEFS_HPP_ */
