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

#ifndef EXAMPLES_DENOISER_VIDEO_FILTER_HPP_
#define EXAMPLES_DENOISER_VIDEO_FILTER_HPP_

#include <vector>

#include <gam.hpp>
#include <ff/d/gff/utils.hpp>

#include <ff/node.hpp>
#include <ff/farm.hpp>

#include "defs.hpp"
#include "seq_detect.hpp"
#ifdef CUDA_RESTORE
#include "cuda_restore.hpp"
#else
#include "seq_restore.hpp"
#endif

/*
 * 1. convert the frame from GAM-friendly to internal representation
 * 2. detect salt-and-pepper noise
 */
class Dispatch: public ff::ff_node
{
public:
    void* svc(void *t)
    {
        auto t0 = gff::hires_timer_ull();

        /* prepare a task */
        filter_task_t *ft = gam::NEW<filter_task_t>();
        ft->ifp = (internal_frame_t *) t;
        ft->ofp = gam::NEW<internal_frame_t>();
        memcpy(ft->ofp, ft->ifp, FRAME_HEIGHT * FRAME_WIDTH);
        ft->noisemap = gam::NEW<noisemap_t>();

        return ft;
    }

private:
    unsigned frame_id = 0;
};

/*
 * convert back to GAM-friendly representation and emit
 */
class Collect: public ff::ff_node
{
    void* svc(void *t)
    {
        filter_task_t *ft = (filter_task_t *) t;
        internal_frame_t *ofp = ft->ofp;

        /* cleanup */
        gam::DELETE(ft->ifp);
        gam::DELETE(ft->noisemap);
        gam::DELETE(ft);

        /* serialize */
        auto sfp = gam::make_private<serial_frame_t>();
        auto local_sfp = sfp.local();
        serialize_frame(local_sfp.get(), ofp);

        /* delete internal representation */
        gam::DELETE(ofp);

        /* emit as gam pointer */
        w2c.emit(gam::private_ptr<serial_frame_t>(std::move(local_sfp)));

        return GO_ON;
    }
};

/*
 * Embeds the pipeline (as accelerator) into gff processor logic
 */
template<typename Comm>
class VideoFilterLogic
{
public:
    VideoFilterLogic()
    {
        /* create stages */
        dispatch = new Dispatch();
        for (unsigned i = 0; i < MWORKERS; ++i)
        {
            detectors.push_back(new Detect());
            restorers.push_back(new Restore());
        }
        collect = new Collect();

        /* create worker-pipelines */
        for (unsigned i = 0; i < MWORKERS; ++i)
        {
            auto p = new ff::ff_pipeline();
            workers.push_back(p);
            p->add_stage(detectors[i]);
            p->add_stage(restorers[i]);
        }

        /* create the farm accelerator */
        accelerator = new ff::ff_ofarm(true /* accelerator */);
        accelerator->setEmitterF(dispatch);
        accelerator->add_workers(workers);
        accelerator->setCollectorF(collect);
    }

    VideoFilterLogic(VideoFilterLogic<Comm> && r)
    {
        dispatch = r.dispatch;
        detectors = std::move(r.detectors);
        restorers = std::move(r.restorers);
        workers = std::move(r.workers);
        collect = r.collect;
        accelerator = r.accelerator;

        r.dispatch = nullptr;
        r.collect = nullptr;
        r.accelerator = nullptr;
    }

    ~VideoFilterLogic()
    {
        if (accelerator)
        delete accelerator;
        for (std::vector<ff::ff_node>::size_type i = 0; i < workers.size(); ++i)
        {
            delete detectors[i];
            delete restorers[i];
            delete workers[i];
        }
        if (dispatch)
        delete dispatch;
        if (collect)
        delete collect;
    }

    gff::token_t
    svc(gam::private_ptr<serial_frame_t> &in, Comm &)
    {
        /* convert the frame to GAM-friendly representation */
        internal_frame_t *ifp = gam::NEW<internal_frame_t>();
        deserialize_frame(ifp, in.local().get());

        /* inject to the pipeline */
        accelerator->offload(ifp);

        return gff::go_on;
    }

    void svc_init()
    {
        /* fire the pipeline accelerator */
        accelerator->run();
    }

    void svc_end()
    {
        /* wait for the pipeline to complete */
        accelerator->offload(EOS);
        accelerator->wait();

        printf("### worker execution times ###\n");
        for (std::vector<ff::ff_node>::size_type i = 0; i < workers.size(); ++i)
        {
            printf("[%lu] detect  = %f s\n", i, detectors[i]->d_detect.count());
            printf("[%lu] restore = %f s\n", i, restorers[i]->svc_time());
        }
        printf("###\n");
    }

private:
    Dispatch *dispatch;
    std::vector<Detect *> detectors;
    std::vector<Restore *> restorers;
    std::vector<ff::ff_node *> workers;
    Collect *collect;
    ff::ff_ofarm *accelerator;
    unsigned received_frames = 0;
}
;

#endif /* EXAMPLES_DENOISER_VIDEO_SOURCE_HPP_ */
