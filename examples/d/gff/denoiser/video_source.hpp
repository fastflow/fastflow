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
 * VideoSource.hpp
 *
 *  Created on: Jun 9, 2017
 *      Author: drocco
 */

#ifndef EXAMPLES_DENOISER_VIDEO_SOURCE_HPP_
#define EXAMPLES_DENOISER_VIDEO_SOURCE_HPP_

#include <ff/node.hpp>
#include <ff/pipeline.hpp>

#include <gam.hpp>
#include <ff/d/gff/utils.hpp>

#include "defs.hpp"

/*
 * Emulates reading a noisy video stream from file or camera
 */
template<typename Comm>
class VideoSourceLogic {
public:

    gff::token_t svc(Comm &c)
    {
        if (emitted_frames == NFRAMES)
            return gff::eos;

        /* convert a frame to GAM-friendly representation */
        auto sfp = gam::make_private<serial_frame_t>();
        auto local_sfp = sfp.local();
        internal_frame_t *ifp = &frame_pool[emitted_frames % frame_pool.size()];
        serialize_frame(local_sfp.get(), ifp);

        /* emit as gam pointer */
        c.emit(gam::private_ptr<serial_frame_t>(std::move(local_sfp)));

        ++emitted_frames;

        return gff::go_on;
    }

    void svc_init()
    {
        /* print parameters in a single line */
        printf("fw\tfh\t%%n\tcyc\tfms\tnw\tmw\n");
        printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", //
                FRAME_WIDTH, FRAME_HEIGHT, NOISE, CYCLES, NFRAMES, //
                NWORKERS, MWORKERS);
    }

    void svc_end()
    {
    }

private:
    unsigned emitted_frames = 0;

    gff::duration_t d_business, d_serialize, d_emit, d_emit_max;
    unsigned d_emit_max_index = 0;
};

#endif /* EXAMPLES_DENOISER_VIDEO_SOURCE_HPP_ */
