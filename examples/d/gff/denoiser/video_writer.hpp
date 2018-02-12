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

#ifndef EXAMPLES_DENOISER_VIDEO_WRITER_HPP_
#define EXAMPLES_DENOISER_VIDEO_WRITER_HPP_

#include <gam.hpp>

#include "defs.hpp"

#ifdef VERIFY
#include "kernel_cpp/pow_table.hpp"
#endif

/*
 * Emulates writing frames to file or display
 */
class VideoWriterLogic {
public:
#ifdef VERIFY
    VideoWriterLogic()
            : noise_dist(0, 99), sp_dist(0, 1), pixel_dist(0, 255), pt(ALFA)
    {
    }
#endif

    gff::token_t svc(gam::private_ptr<serial_frame_t> &in)
    {
        /* convert to internal representation */
        auto local_sfp = in.local();
        internal_frame_t ifp;
        deserialize_frame(&ifp, local_sfp.get());

        printf("[frame # %d]\tchecksum = %llu\n", frame_id, checksum(ifp));

#ifdef CHECK
        print_sp_frame(frame_id, ifp);
#endif

        ++frame_id;

#ifdef VERIFY
        internal_frame_t v_ifp, v_ofp;
        noisemap_t v_nm;
        /* generate a random frame  */
        for (unsigned i = 0; i < FRAME_HEIGHT * FRAME_WIDTH; ++i)
            v_ifp.data[i] = rng();

        /* add noise */
        for (unsigned i = 0; i < FRAME_HEIGHT * FRAME_WIDTH; ++i)
            if (noise_dist(rng) < NOISE)
                v_ifp.data[i] = sp_dist(rng) * 255;

        /* apply the detect-restore filter */
        sp_detect(&v_ifp, &v_nm);
        restore(&v_ifp, &v_ofp, &v_nm, pt);

        /* compare with the incoming restored frame */
        assert(checksum(ifp) == checksum(v_ifp));
#endif

        return gff::go_on;
    }

    void svc_init()
    {
    }

    void svc_end()
    {
    }

private:
    unsigned frame_id = 0;
#ifdef VERIFY
    std::mt19937 rng;
    std::uniform_int_distribution<int> noise_dist, sp_dist;
    std::uniform_int_distribution<unsigned char> pixel_dist;
    pow_table pt;
#endif
};

#endif /* EXAMPLES_DENOISER_VIDEO_SOURCE_HPP_ */
