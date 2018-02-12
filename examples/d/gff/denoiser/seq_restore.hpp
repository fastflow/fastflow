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
 * @file        seq_restore.hpp
 * @author      Maurizio Drocco
 * 
 */
#ifndef EXAMPLES_DENOISER_SEQ_RESTORE_HPP_
#define EXAMPLES_DENOISER_SEQ_RESTORE_HPP_

#include <ff/node.hpp>

#include "defs.hpp"

#include "kernel_cpp/fuy.hpp"
#include "kernel_cpp/pow_table.hpp"

/*
 * restore the frame
 */
class Restore: public ff::ff_node {
public:
    Restore()
            : pt(ALFA)
    {
    }

    void* svc(void *t)
    {
        filter_task_t *ft = (filter_task_t *) t;

        /* restore */
        restore(ft->ifp, ft->ofp, ft->noisemap, pt);
        printf("[frame # %d]\tchecksum = %llu\tcycles = %d\n", //
                restored_frames, checksum(*ft->ofp), CYCLES);
        ++restored_frames;

        return ft;
    }

    double svc_time() {
        return wffTime() / 1000;
    }

private:
    pow_table pt;
    unsigned restored_frames = 0;
};

#endif /* EXAMPLES_DENOISER_SEQ_RESTORE_HPP_ */
