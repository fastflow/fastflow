#ifndef __CUDA_DENOISER_HPP__
#define __CUDA_DENOISER_HPP__

#include <cuda_dispatcher.hpp>
#include "utils.hpp"
//#include "pow_table.hpp"
#include <convergence.hpp>

template <typename T>
class Cuda_denoiser {
public:
  Cuda_denoiser(
		Bitmap<T> *bmp,
		float alpha,
		float beta,
		vector<noisy<T> > &noisy_set,
		bmp_size_t height,
		bmp_size_t width,
		bool verbose
		)
    :
    bmp(bmp),
    alpha(alpha), beta(beta),
    cuda_dispatcher_ptr(NULL),
    set(noisy_set),
    height(height), width(width)
  {
    get_cuda_env(corecount, false);
  }

  ~Cuda_denoiser() {
    cuda_dispatcher_ptr->releaseMemory();
  }



  residual_t svc() {
    cuda_dispatcher_ptr->run();
#ifndef AVG_TERMINATION
    return cuda_dispatcher_ptr->MaxResidual();
#else
    return cuda_dispatcher_ptr->AvgResidual();
#endif
  }



  void svc_init() {
    cuda_dispatcher_ptr = new CUDADispatcher<T>();
    // Set algorithm parameters
    cuda_dispatcher_ptr->alg_parameters.alpha = alpha;
    cuda_dispatcher_ptr->alg_parameters.beta = beta; //beta;
    cuda_dispatcher_ptr->initDevice();

    //set threads_per_block such that there is at last one block per core
    threads_per_block = 256; //max tpb density
    grain = 1;
    int n_noisy = set.size();
    while (n_noisy/(grain*threads_per_block) < corecount)
      threads_per_block/=2;

    if (threads_per_block*grain> 1024)
      cerr << "Threads_per_block*grain is high, this may lead to overcome shared memory limits on some device (apps won't work correctly)\n";
    if ((!(IsPowerofTwo(grain))) || (!(IsPowerofTwo(threads_per_block)))) 
      cerr << "Block dim and Grain ought to be Power of 2\n";

    cuda_dispatcher_ptr->initMemory(n_noisy, height, width, threads_per_block, grain);
    cuda_dispatcher_ptr->host2device(set, n_noisy, *bmp, height, width);
  }



  void svc_end() {
    cuda_dispatcher_ptr->device2host(*bmp, height, width);
  }



private:
  Bitmap<T> *bmp;
  double alpha;
  double beta;
  CUDADispatcher<T> * cuda_dispatcher_ptr;
  vector<noisy<T> > &set;
  int height, width;
  int corecount, threads_per_block, grain;
};


#endif
