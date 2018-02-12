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
 * @file        farm_components.hpp
 * @author      Maurizio Drocco
 * 
 */
#ifndef EXAMPLES_STOCK_MARKET_FARM_COMPONENTS_HPP_
#define EXAMPLES_STOCK_MARKET_FARM_COMPONENTS_HPP_

#include <ff/d/gff/gff.hpp>

#include "defs.hpp"
#include "option_source.hpp"
#include "option_filter.hpp"
#include "price_writer.hpp"

/*
 ***************************************************************************
 *
 * farm components
 *
 ***************************************************************************
 */
typedef gff::Source<gff::RoundRobinSwitch, //
        gam::private_ptr<option_batch_t>, //
        OptionSourceLogic<gff::RoundRobinSwitch>> OptionSource;

typedef gff::Sink<gff::RoundRobinMerge, //
        gam::private_ptr<option_batch_t>, //
        PriceWriterLogic> PriceWriter;

typedef gff::Filter<gff::RoundRobinSwitch, gff::RoundRobinMerge, //
        gam::private_ptr<option_batch_t>, gam::private_ptr<option_batch_t>, //
        OptionFilterLogic<gff::RoundRobinMerge>> OptionFilter;

#endif /* EXAMPLES_STOCK_MARKET_FARM_COMPONENTS_HPP_ */
