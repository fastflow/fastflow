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

#ifndef EXAMPLES_DENOISER_DEFS_HPP_
#define EXAMPLES_DENOISER_DEFS_HPP_

#include <ff/d/gff/gff.hpp>

#include "config.hpp"
#include "kernel_cpp/pow_table.hpp"
#include "kernel_cpp/fuy.hpp"

/*
 * communicators
 */
gff::RoundRobinSwitch e2w;
gff::RoundRobinMerge w2c;

/*
 * application types
 */
struct serial_frame_t {
    unsigned char data[FRAME_HEIGHT * FRAME_WIDTH];
};

typedef serial_frame_t internal_frame_t;

struct noisemap_t {
    int data[FRAME_HEIGHT * FRAME_WIDTH];
    unsigned int indexes[FRAME_HEIGHT * FRAME_WIDTH];
    unsigned int n_noisy = 0;
};

/*
 * conversion functions for GAM-friendly representation
 */
void serialize_frame(serial_frame_t *dst, internal_frame_t *src)
{
    memcpy(dst, src, FRAME_HEIGHT * FRAME_WIDTH);
}

void deserialize_frame(internal_frame_t *dst, serial_frame_t *src)
{
    memcpy(dst, src, FRAME_HEIGHT * FRAME_WIDTH);
}

/*
 * farm tasks
 */
struct filter_task_t {
    internal_frame_t *ifp, *ofp;
    noisemap_t *noisemap;
};

/*
 * source functions
 */
#include <random>

std::vector<internal_frame_t> frame_pool;

void frame_pool_init()
{
    std::mt19937 rng;
    std::uniform_int_distribution<int> noise_dist(0, 99), sp_dist(0, 1);
    std::uniform_int_distribution<unsigned char> pixel_dist(0, 255);

    for (unsigned fi = 0; fi < 16; ++fi)
    {
        frame_pool.push_back(internal_frame_t());
        internal_frame_t *ifp = &frame_pool[fi];
        for (unsigned i = 0; i < FRAME_HEIGHT * FRAME_WIDTH; ++i)
            ifp->data[i] = pixel_dist(rng);

        /* add noise */
        for (unsigned i = 0; i < FRAME_HEIGHT * FRAME_WIDTH; ++i)
            if (noise_dist(rng) < NOISE)
                ifp->data[i] = sp_dist(rng) * 255;
    }
}

/*
 * detect functions
 */
constexpr unsigned MAX_WINDOW_SIZE = 39 * 2 + 1;

int ctrl_px(unsigned char *im, unsigned pos, //
        unsigned char *a, unsigned int w)
{
    unsigned yy = pos / FRAME_WIDTH;
    unsigned xx = pos % FRAME_WIDTH;

    unsigned int cw(w / 2);
    int n = 0;
    for (unsigned int r = 0; r < w; r++) //rows
        for (unsigned int c = 0; c < w; c++) //columns
            if (((xx + c) >= cw) && ((xx + c) < (FRAME_WIDTH + cw))
                    && ((yy + r) >= cw) && ((yy + r) < (FRAME_HEIGHT + cw)))
                a[n++] = im[(yy + r - cw) * FRAME_WIDTH + xx + c - cw];
    return n;
}

inline bool is_noisy(unsigned char *im, unsigned pos, unsigned int w_max)
{
    unsigned char center = im[pos];
    if (center == 255 || center == 0)
    {
        unsigned char c_array[MAX_WINDOW_SIZE * MAX_WINDOW_SIZE];
        for (unsigned int ws = 3; ws < w_max; ws += 2)
        {
            int ac = ctrl_px(im, pos, c_array, ws);
            sort(c_array, c_array + ac);
            unsigned char min_cp = c_array[0];
            unsigned char med_cp = c_array[ac / 2];
            unsigned char max_cp = c_array[ac - 1];
            if ((!(min_cp == med_cp && med_cp == max_cp)) || (ws > w_max)) //not homogeneous zone
                if (center == min_cp || center == max_cp || ws > w_max)
                    return true; //noisy
        }
    }
    return false; //not noisy
}

void sp_detect(internal_frame_t *fp, noisemap_t *np)
{
    for (unsigned i = 0; i < FRAME_HEIGHT * FRAME_WIDTH; ++i)
    {
        if (is_noisy(fp->data, i, 39))
        {
            np->data[i] = fp->data[i];
            np->indexes[np->n_noisy++] = i;
        }
        else
        {
            np->data[i] = -1;
        }

    }
}

/*
 * restore functions
 */
inline unsigned char restorePixel(unsigned char *in, unsigned int x, int *n, //
        pow_table &pt)
{
    return fuy(in, x, n, FRAME_WIDTH, FRAME_HEIGHT, ALFA, BETA, pt);
}

void restore(internal_frame_t *ifp, internal_frame_t *ofp, //
        noisemap_t *np, pow_table &pt)
{
    unsigned char *in = ifp->data;
    unsigned char *out = ofp->data;
    memcpy(out, in, FRAME_HEIGHT * FRAME_WIDTH);

    for (unsigned c = 0; c < CYCLES; ++c)
    {
        //restore
        for (unsigned int i = 0; i < np->n_noisy; ++i)
        {
            unsigned int x = np->indexes[i];
            out[x] = restorePixel(in, x, np->data, pt);
        }
        memcpy(in, out, FRAME_HEIGHT * FRAME_WIDTH);
    }
}

/*
 * misc
 */
unsigned long long checksum(internal_frame_t &f)
{
    unsigned long long res = 0;
    for (unsigned i = 0; i < FRAME_HEIGHT * FRAME_WIDTH; ++i)
        res += f.data[i];
    return res;
}

void print_sp_frame(unsigned id, internal_frame_t &f, noisemap_t &n)
{
    float noise = (float) n.n_noisy / (FRAME_HEIGHT * FRAME_WIDTH);
    printf("[frame # %d]\tchecksum = %llu\t%%-noise = %f\n", //
            id, checksum(f), noise * 100);
}

void print_sp_frame(unsigned id, internal_frame_t &f)
{
    noisemap_t n;
    sp_detect(&f, &n);
    print_sp_frame(id, f, n);
}

#endif /* EXAMPLES_DENOISER_DEFS_HPP_ */
