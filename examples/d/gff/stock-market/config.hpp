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
 * config.hpp
 *
 *  Created on: Jun 9, 2017
 *      Author: drocco
 */

#ifndef EXAMPLES_STOCK_MARKET_BATCH_CONFIG_HPP_
#define EXAMPLES_STOCK_MARKET_BATCH_CONFIG_HPP_

#include <string>

const std::string in_fname = "./sampledata/stock_options_64K.txt";

#ifndef STREAM_LEN
#define STREAM_LEN (1 << 20)
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 512
#endif

#ifndef REBUTTAL
#define REBUTTAL 100
#endif

#ifndef NWORKERS
#define NWORKERS 1
#endif

#endif /* EXAMPLES_STOCK_MARKET_BATCH_CONFIG_HPP_ */
