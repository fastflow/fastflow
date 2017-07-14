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
/* Author: Maurizio Drocco
 *         drocco@di.unito.it
 */

#if !defined(FF_CUDA)
#define FF_CUDA
#endif

#include <ff/farm.hpp>
#include <ff/stencilReduceCUDA.hpp>
using namespace ff;

#include <iostream>
using namespace std;

FFMAPFUNC(mapF, unsigned int, in, return in + 1;);

// this is global just to keep things simple
size_t inputsize = 2048;

class cudaTask: public baseCUDATask<unsigned int, unsigned int, unsigned char,
		unsigned char, int, char, char> {
public:
	void setTask(void* t) {
		if (t) {
			cudaTask *t_ = (cudaTask *) t;
			setInPtr(t_->in);
			setOutPtr(t_->in);
			//setOutPtr(t_->out);
			setSizeIn(inputsize);
		}
	}

	unsigned int *in, *out;
};

int main(int argc, char * argv[]) {
	if (argc > 1) inputsize = atoi(argv[1]);
	printf("using arraysize = %lu\n", inputsize);

	cudaTask *task = new cudaTask();
	task->in = new unsigned int[inputsize];
	for (size_t j = 0; j < inputsize; ++j)
		task->in[j] = j;



	FFMAPCUDA(cudaTask, mapF) *myMap = new FFMAPCUDA(cudaTask, mapF)(*task);
	myMap->run_and_wait_end();

//#ifdef CHECK
	for (size_t i = 0; i < inputsize; ++i) {
		if (task->in[i] != (i + 1)) {
			printf("ERROR\n");
			return 1;
		}
	}
//#endif

	delete[] task->in;

	printf("DONE\n");
	return 0;
}
