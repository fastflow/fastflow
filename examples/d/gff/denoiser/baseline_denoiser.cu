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
 * @file        baseline_denoiser.cpp
 * @brief       baseline salt-and-pepper restoration filter, based on CUDA
 * @author      Maurizio Drocco
 *
 */

#include <ff/pipeline.hpp>
#include <ff/node.hpp>

#include "defs.hpp"
#include "seq_detect.hpp"
#include "cuda_restore.hpp"

class VideoSource: public ff::ff_node
{
public:
    VideoSource() :
            noise_dist(0, 99), sp_dist(0, 1), pixel_dist(0, 255)
    {
    }

    void * svc(void *)
    {
        if (emitted_frames == NFRAMES)
            return EOS;

        /* prepare a task */
        filter_task_t *ft = new filter_task_t();
        ft->ifp = new internal_frame_t();
        ft->ofp = new internal_frame_t();
        ft->noisemap = new noisemap_t();

        internal_frame_t *ifp = &frame_pool[emitted_frames % frame_pool.size()];
        memcpy(ft->ifp, ifp, FRAME_HEIGHT * FRAME_WIDTH);
        memcpy(ft->ofp, ft->ifp, FRAME_HEIGHT * FRAME_WIDTH);

        /* emit */
        ++emitted_frames;
        return ft;
    }

private:
    std::mt19937 rng;
    std::uniform_int_distribution<int> noise_dist, sp_dist;
    std::uniform_int_distribution<unsigned char> pixel_dist;
    unsigned emitted_frames = 0;
};

class FrameCollector: public ff::ff_node
{
public:
    void * svc(void *t)
    {
        filter_task_t *ft = (filter_task_t *) t;

        printf("[frame # %d]\tchecksum = %llu\n", frame_id, checksum(*ft->ofp));

#ifdef CHECK
        print_sp_frame(frame_id, *ft->ofp);
#endif

        ++frame_id;

        /* clean up */
        delete ft->ifp;
        delete ft->ofp;
        delete ft->noisemap;

        return GO_ON;
    }

private:
    unsigned frame_id = 0;
};

/*
 *******************************************************************************
 *
 * main
 *
 *******************************************************************************
 */
int main(int argc, char * argv[])
{
    frame_pool_init();
    
    VideoSource vs;
    Detect d;
    Restore r;
    FrameCollector fc;

    ff::ff_pipeline p;
    p.add_stage(&vs);
    p.add_stage(&d);
    p.add_stage(&r);
    p.add_stage(&fc);
    p.run_and_wait_end();

    /* print parameters and measure in a single line */
    printf("fw\tfh\t%%n\tcyc\tfms\tnw\tmw\ttime\n");
    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%f\n", //
            FRAME_WIDTH, FRAME_HEIGHT, NOISE, CYCLES, NFRAMES, //
            NWORKERS, MWORKERS, p.ffTime() / 1000);

    return 0;
}
