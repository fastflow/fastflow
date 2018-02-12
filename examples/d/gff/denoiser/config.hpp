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

#ifndef EXAMPLES_DENOISER_CONFIG_HPP_
#define EXAMPLES_DENOISER_CONFIG_HPP_

#ifndef FRAME_WIDTH
#define FRAME_WIDTH 320
#endif

#ifndef FRAME_HEIGHT
#define FRAME_HEIGHT 240
#endif

#ifndef NOISE
#define NOISE 50
#endif

#ifndef NFRAMES
#define NFRAMES 8
#endif

#ifndef CYCLES
#define CYCLES 20
#endif

#define ALFA 1.3f
#define BETA 5.0f

#ifndef NWORKERS
#define NWORKERS 1
#endif

#ifndef MWORKERS
#define MWORKERS 1
#endif

//#define CHECK //checks if restoration is doing something

//#define VERIFY //checks if restoration is doing something correct

#endif /* EXAMPLES_DENOISER_CONFIG_HPP_ */
