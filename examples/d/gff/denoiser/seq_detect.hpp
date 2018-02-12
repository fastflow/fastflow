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
 * @author      Maurizio Drocco
 * 
 */
#ifndef EXAMPLES_DENOISER_SEQ_DETECT_HPP_
#define EXAMPLES_DENOISER_SEQ_DETECT_HPP_

#include <ff/node.hpp>

#include <ff/d/gff/utils.hpp>

#include "defs.hpp"

/*
 * detect noise for the frame
 */
class Detect: public ff::ff_node
{
public:
    void* svc(void *t)
    {
        filter_task_t *ft = (filter_task_t *) t;

        auto t0 = gff::hires_timer_ull();

        /* detect */
        sp_detect(ft->ifp, ft->noisemap);

        d_detect += gff::time_diff(t0, gff::hires_timer_ull());

        print_sp_frame(detected_frames, *ft->ifp, *ft->noisemap);

        ++detected_frames;

        return ft;
    }

    gff::duration_t d_detect;

private:
    unsigned detected_frames = 0;
};

#endif /* EXAMPLES_DENOISER_SEQ_DETECT_HPP_ */
