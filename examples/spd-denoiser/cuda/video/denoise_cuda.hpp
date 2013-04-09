#ifndef __DENOISE_CUDA_HPP__
#define __DENOISE_CUDA_HPP__

#include "ff_accel_video.hpp"
#include <ff/node.hpp>
#include <iostream>
//#include "cuda_utils/cutil.h"
//#include "cuda_utils/cutil_inline_runtime.h"
#include <cuda_runtime_api.h>
//#include <ff/pipeline.hpp>
//#include "taskTypes.hpp"
//#include "utils.h"
//#include "pow_table.hpp"
//#include "fuy.hpp"
#include "convergence.hpp"
//#include "denoise_cuda.h"
// ---
#include "cuda_dispatcher.hpp"
#include "time_interval.h"

template <typename T>
class Denoise_cuda : public ff::ff_node {
public:
  Denoise_cuda(double alpha, double beta) : alpha(alpha), beta(beta), cuda_dispatcher_ptr(NULL) {
    try
      {
	int deviceCount = 0;
	cutilSafeCall(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0)
	  std::cerr << "There is no device supporting CUDA\n";
	else
	  std::cerr << "\nFound " << deviceCount << " CUDA Capable device(s)\n";
	int dev, driverVersion = 0, runtimeVersion = 0;     
	for (dev = 0; dev < deviceCount; ++dev) {
	  cudaDeviceProp deviceProp;
	  cudaGetDeviceProperties(&deviceProp, dev);
		  
	  printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
		
#if CUDART_VERSION >= 2020
	  // Console log
	  cudaDriverGetVersion(&driverVersion);
	  cudaRuntimeGetVersion(&runtimeVersion);
	  printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);
#endif
	  printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
	  char msg[256];
	  sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",(float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
	  printf("%s",msg);
#if CUDART_VERSION >= 2000
	  printf("  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n", 
		 deviceProp.multiProcessorCount,
		 _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		 _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
#endif
	  printf("  GPU Clock Speed:                               %.2f GHz\n", deviceProp.clockRate * 1e-6f);

	  corecount = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;

	}
      }
    catch(string exc)
      {
	cout << exc << endl;
      }
    catch(...)
      {
	cout << "Some other exception..." << endl;
      }
  }



  void * svc(void * task) {
    noisy_img_task<T> *t = (noisy_img_task<T> *)task;
    vector<noisy<T> > &set = *(t->the_noisy_set);	
    int height = t->bmp->height(), width = t->bmp->width();
    TimeInterval partial,total;
    total.tic();
    // ---
    int threads_per_block = 256;
    int grain = 1; 
    if (set.size()==0) {
      cerr << "Skipping denoising\n";
      return task;
    }
    while ((set.size()/(grain*threads_per_block)<corecount) && 
	   (threads_per_block>2))
      threads_per_block/=2;
    //cerr << "Threads per block " << threads_per_block << "\n";
		  
    if (threads_per_block*grain> 1024)
      cerr << "Threads_per_block*grain is high, this may lead to overcome shared memory limits on some device (apps won't work correctly)\n";
    if ((!(IsPowerofTwo(grain))) || (!(IsPowerofTwo(threads_per_block)))) 
      cerr << "Block dim and Grain ought to be Power of 2\n";

#ifdef TIME
    long t_rec = get_usec_from(0);
#endif	
  
    cuda_dispatcher_ptr->initMemory(set.size(), height, width, threads_per_block, grain);
    cuda_dispatcher_ptr->host2device(set, set.size(), *(t->bmp), height, width);
 
    // Algorithm parameters
    int max_cycles = 100;
    // Variables for termination criteria evaluation
    int current_cycle;
    residual_t current_max_residual = INT_MAX;
    residual_t old_max_residual = INT_MAX;
    residual_t delta_max_residual = INT_MAX;
    // Main processing loop
    for(current_cycle=0; current_cycle<max_cycles && (delta_max_residual>1 || current_max_residual==255); current_cycle++) {
      //cuda_dispatcher_ptr->mapToDevice(cuda_image);
      // Run iteration
      partial.tic();
      //cout << "\tRunning on device... " << flush;
      cuda_dispatcher_ptr->run();
      //cout << "Iteration " << current_cycle << ": " 
      //	 << partial.toc()*1000 << " ms" << endl;
      old_max_residual = current_max_residual;
      partial.tic();
      cudaDeviceSynchronize();
      // Read cluster max residual
      current_max_residual = cuda_dispatcher_ptr->MaxResidual();
      /*
	if(current_cycle == 0)
	std::cout << "[cuda-int]: " << current_max_residual << endl;
      */
      //cout << "\tResidual copy: " << partial.toc()*1000 << " ms\n";
      //cout << "\tMax residual: " << current_max_residual << endl;
      delta_max_residual = _ABS(current_max_residual - old_max_residual);
      //cout << "done (" << partial.toc() << " s)" << endl;	
    }
  
    partial.tic();
  
    /*
      for(int i=0; i<height; i++) {
      for(int j=0; j<width; j++) {
      t->bmp->set(j, i, cuda_image.get(i,j).grey);
      }
      }
    */
    cuda_dispatcher_ptr->device2host(*(t->bmp), height, width);

#ifdef DEBUG
    cerr << "Copy image form device " << partial.toc()*1000 << " (ms)\n";
    TimeInterval rel;
    rel.tic();
#endif
    cuda_dispatcher_ptr->releaseMemory();
#ifdef DEBUG
    cerr << "Release memory " << rel.toc()*1000 << " (ms)\n";
#endif 
	
#ifdef TIME
    t_rec = get_usec_from(t_rec)/1000;
    cerr << "Denoising time :" << t_rec << " (ms) Cuda time: " << total.toc()*1000 << " (ms)"  << " Num. iterations: " << current_cycle << endl;
#endif

    return task;
  }



  int svc_init() {
    cuda_dispatcher_ptr = new CUDADispatcher<T>();
    // Set algorithm parameters
    cuda_dispatcher_ptr->alg_parameters.alpha = alpha;
    cuda_dispatcher_ptr->alg_parameters.beta = beta; //beta;
    cuda_dispatcher_ptr->initDevice();
    return 0;
  }



private:
  double alpha;
  double beta;
  CUDADispatcher<T> * cuda_dispatcher_ptr;
  int corecount;
};

#endif
