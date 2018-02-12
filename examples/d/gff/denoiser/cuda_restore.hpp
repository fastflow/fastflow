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
 * @file        cuda_restore.hpp
 * @author      Maurizio Drocco
 * 
 */
#ifndef EXAMPLES_DENOISER_CUDA_RESTORE_HPP_
#define EXAMPLES_DENOISER_CUDA_RESTORE_HPP_

#ifndef FF_CUDA
#error "FF_CUDA flag must be passed to the compiler";
#endif

#include "ff-patch/stencilReduceCUDA.hpp"

#include "defs.hpp"

float residuals[FRAME_HEIGHT * FRAME_WIDTH];

struct mapF
{
    __device__ float K(unsigned int idx, //argument
            unsigned char *in, //env1
            unsigned char *out, //env2
            int *noisymap, //env3
            float *params, //env4
            unsigned int *bitmap_size, //env5
            float *diff //env6
            )
    {
        float alfa = params[0];
        float beta = params[1];
        unsigned int h = bitmap_size[0];
        unsigned int w = bitmap_size[1];
        //get the pixel and the four closest neighbors
        //(with replication-padding)
        unsigned char pixel = in[idx];
        //up
        int idx_neighbor = idx - w * (idx >= w);
        unsigned char up_val = in[idx_neighbor];
        unsigned char up_noisy = (noisymap[idx_neighbor] >= 0);
        //down
        idx_neighbor = idx + w * (idx < ((h - 1) * w));
        unsigned char down_val = in[idx_neighbor];
        unsigned char down_noisy = (noisymap[idx_neighbor] >= 0);
        //left
        idx_neighbor = idx - ((idx % w) > 0);
        unsigned char left_val = in[idx_neighbor];
        unsigned char left_noisy = (noisymap[idx_neighbor] >= 0);
        //right
        idx_neighbor = idx + ((idx % w) < (w - 1));
        unsigned char right_val = in[idx_neighbor];
        unsigned char right_noisy = (noisymap[idx_neighbor] >= 0);
        //compute the correction
        unsigned char u;
        float S;
        float Fu;
        float u_min = 0.0f;
        float Fu_prec = FLT_MAX;
        float beta_ = beta / 2;
        for (int uu = 0; uu < 256; ++uu)
        {
            u = (unsigned char) uu;
            Fu = 0.0f;
            S = 0.0f;
            S += (float) (2 - up_noisy) * __powf(abs(uu - (int) up_val), alfa);
            S += (float) (2 - down_noisy)
                    * __powf(abs(uu - (int) down_val), alfa);
            S += (float) (2 - left_noisy)
                    * __powf(abs(uu - (int) left_val), alfa);
            S += (float) (2 - right_noisy)
                    * __powf(abs(uu - (int) right_val), alfa);
            Fu += abs((float) u - (float) pixel) + (beta_) * S;
            if (Fu < Fu_prec)
            {
                u_min = u;
                Fu_prec = Fu;
            }
        }
        out[idx] = (unsigned char) (u_min + 0.5f); //round
        return 0;
    }
};

struct devicereduceF
{
    __device__ unsigned int K(unsigned int x, unsigned int y, //
            void *, void *, void *, void *, void *, void *)
    {
        return x + y;
    }
};

struct hostreduceF
{
    unsigned int K(unsigned int x, unsigned int y, //
            void *, void *, void *, void *, void *, void *)
    {
        return x + y;
    }
};

class cudaTask: public ff::baseCUDATask<unsigned int, float, //
        unsigned char, unsigned char, int, float, unsigned int, float>
{
public:
    void setTask(void* t)
    {
        if (t)
        {
            filter_task_t * task_ = (filter_task_t *) t;
            in = task_->ifp->data;
            out = task_->ofp->data;
            noisy = task_->noisemap->indexes;
            n_noisy = task_->noisemap->n_noisy;
            height = FRAME_HEIGHT;
            width = FRAME_WIDTH;
            this->setInPtr(noisy);
            this->setSizeIn(n_noisy);
            this->setOutPtr(residuals); //unused
            this->setEnv1Ptr(in);
            this->setSizeEnv1(height * width);
            this->setEnv2Ptr(out);
            this->setSizeEnv2(height * width);
            this->setEnv3Ptr(task_->noisemap->data);
            this->setSizeEnv3(height * width);
            params[0] = ALFA;
            params[1] = BETA;
            this->setEnv4Ptr(params);
            this->setSizeEnv4(2);
            sizes[0] = height;
            sizes[1] = width;
            this->setEnv5Ptr(sizes);
            this->setSizeEnv5(2);
            this->setEnv6Ptr(nullptr);
            this->setSizeEnv6(0);
        }
    }

#if 0
    void beforeMR()
    {
        internal_frame_t tmp1, tmp2;
        unsigned int in_[FRAME_HEIGHT * FRAME_WIDTH];
        cudaMemcpy(in_, getInDevicePtr(), //
                getSizeIn() * sizeof(unsigned int), //
                cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp1.data, getEnv1DevicePtr(), //
                height * width * sizeof(unsigned char), //
                cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp2.data, getEnv2DevicePtr(), //
                height * width * sizeof(unsigned char), //
                cudaMemcpyDeviceToHost);
        printf("********\nbeforeMR in @ %p\n", getInDevicePtr());
        printf("#noisy = %u\n", getSizeIn());
        printf("beforeMR env1 @ %p\n", getEnv1DevicePtr());
        print_sp_frame(0, tmp1);
        printf("beforeMR env2 @ %p\n", getEnv2DevicePtr());
        print_sp_frame(0, tmp2);
    }

    void afterMR(void*)
    {
        internal_frame_t tmp1, tmp2;
        cudaMemcpy(tmp1.data, getEnv1DevicePtr(), //
                height * width * sizeof(unsigned char), //
                cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp2.data, getEnv2DevicePtr(), //
                height * width * sizeof(unsigned char), //
                cudaMemcpyDeviceToHost);
        printf("afterMR env1 @ %p\n", getEnv1DevicePtr());
        print_sp_frame(0, tmp1);
        printf("afterMR env2 @ %p\n", getEnv2DevicePtr());
        print_sp_frame(0, tmp2);
    }
#endif

    void endMR(void*)
    {
        cudaMemcpy(out, getEnv2DevicePtr(), //
                height * width * sizeof(unsigned char), //
                cudaMemcpyDeviceToHost);
    }

    void swap()
    {
        unsigned char *tmp = getEnv1DevicePtr();
        setEnv1DevicePtr(getEnv2DevicePtr());
        setEnv2DevicePtr(tmp);
    }

private:
    float params[2]; //alfa, beta
    unsigned int sizes[2]; //width, height
    unsigned int *noisy, n_noisy, height, width;
    unsigned char *in, *out;
};

typedef ff::ff_stencilReduceCUDA<cudaTask, mapF, devicereduceF, hostreduceF> //
RestoreBase;

class Restore: public RestoreBase
{
public:
    Restore() :
            RestoreBase(CYCLES)
    {
    }

    double svc_time() {
        return wffTime() / 1000;
    }
};

#endif /* EXAMPLES_DENOISER_CUDA_RESTORE_HPP_ */
