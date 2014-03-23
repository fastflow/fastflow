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
/* Author: Massimo Torquati / Guilherme Peretti Pezzi
 *         torquati@di.unipi.it  massimotor@gmail.com  / peretti@di.unito.it
 */


#if !defined(FF_CUDA)
#define FF_CUDA
#endif

#include <ff/mapCUDAManaged.hpp>

#include <cuda_runtime.h>

using namespace ff;

template<typename Tenv, typename Tinout>
class cudaTask: public baseCUDATaskManaged {
public:
    typedef Tenv env_type;
    typedef Tinout inout_type;

    cudaTask():s(0) {}
    cudaTask(Tenv * env, Tinout * in, Tinout * out, size_t size) : baseCUDATaskManaged(env, in, out), s(size) {}

    size_t size() const     { return s;}

protected:
    size_t s;
};

class DenoiseEnv : public ff_cuda_managed{
public:
	DenoiseEnv(float alfa, float beta, int n_noisy) : alfa(alfa), beta(beta), n_noisy(n_noisy){};

	float alfa;
	float beta;
	int n_noisy;
};

FFMAPFUNCMANAGED(mapf, unsigned char, DenoiseEnv*, env, unsigned char, in, return (unsigned char) (env->beta * in));

int main(int argc, char * argv[]) {
	if (argc<2) {
		printf("use %s arraysize\n", argv[0]);
		return -1;
	}

#if defined(__APPLE__) || defined(MACOSX)
    fprintf(stderr, "Unified Memory not currently supported on OS X\n");
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
#endif

	size_t size     =atoi(argv[1]);
	DenoiseEnv * env = new DenoiseEnv(2.1 , 3.4 , size);
	unsigned char * in;
	unsigned char * out;

	cudaMallocManaged(&in, size *sizeof(unsigned char));
	cudaMallocManaged(&out, size *sizeof(unsigned char));

	/* init data */
	for(size_t j=0;j<size;++j) in[j]=j;

//	NEWMAPMANAGED(cudamap, (cudaTask<DenoiseEnv, unsigned char>), mapf, env, in , out, size);

	ff_mapCUDAManaged<cudaTask<DenoiseEnv, unsigned char>, mapf> *cudamap =
			new ff_mapCUDAManaged<cudaTask<DenoiseEnv, unsigned char>, mapf>(new mapf, env, in, out, size) ;

	for(long i=0;i<size;++i)  printf("%d ", in[i]);
	printf(" = INPUT\n");
	printf("Env ( alfa = %f / beta = %f / n_noisy = %d )  \n", env->alfa, env->beta, env->n_noisy);

	cudamap->run_and_wait_end();

	for(long i=0;i<size;++i)  printf("%d ", out[i]);
	printf(" = OUTPUT\n");

	DELETEMAP(cudamap);

	if(env) cudaFree(env);
	if (out) cudaFree(out);
	if (in) cudaFree(in);

	printf("DONE\n");

	return 0;
}
