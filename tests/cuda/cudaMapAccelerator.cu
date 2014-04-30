/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */
/* Author: Massimo Torquati
 *         
 */

#if !defined(FF_CUDA)
#define FF_CUDA
#endif

#include <vector>
#include <ff/pipeline.hpp>
#include <ff/stencilReduceCUDA.hpp>
using namespace ff;

FFMAPFUNC(mapF, unsigned int, in, return in + 1;);

struct cudaTask: public baseCUDATask<unsigned int> {
	void setTask(void* t) {
		if (t) {
			cudaTask *t_ = (cudaTask *) t;
			setInPtr(t_->buffer);
			setOutPtr(t_->buffer);
			setSizeIn(t_->size);
		}
	}

	unsigned int *buffer;
    size_t        size;
};

int main(int argc, char * argv[]) {
	if (argc < 3) {
		printf("use %s arraysize ntasks\n", argv[0]);
		return -1;
	}
	int inputsize = atoi(argv[1]);
    int ntasks    = atoi(argv[2]);

    ff_pipeline pipe(true);
    pipe.add_stage(new FFMAPCUDA(cudaTask, mapF));
	pipe.run_then_freeze();
    pipe.offload(GO_OUT);
    pipe.wait_freezing();

    std::vector<unsigned int> V(inputsize);

    pipe.run_then_freeze();
    for(int i=0;i<ntasks;++i) {

        for (size_t j = 0; j < inputsize; ++j)
            V[j] = i+j;

        cudaTask tmp, *ptmp;
        ptmp = &tmp;
        tmp.buffer = V.data();
        tmp.size   = V.size();


        pipe.offload((void*)ptmp);
        pipe.load_result((void**)&ptmp);

        for (size_t j = 0; j < inputsize; ++j)
            printf("%d ", V[j]);
        printf("\n");
    }
    pipe.offload(GO_OUT);
    pipe.wait_freezing();

	printf("DONE\n");
	return 0;
}
