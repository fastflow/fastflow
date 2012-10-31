// Class for the division of a task among a set of workers

#ifndef _SPD_CUDA_DISPATCHER_HPP_
#define _SPD_CUDA_DISPATCHER_HPP_

//System
#include <control_structures.hpp>
#include <cuda_runtime_api.h>
#include <cuda_utils/cutil.h>
#include <cuda_utils/cutil_inline_runtime.h>
#include <sstream>
#include <assert.h>

//Own
#include <definitions.h>
#include <bitmap.hpp>
#include <cuda_definitions.h>
#include <convergence.hpp>
#include "cuda_kernel.h"
#include <utils.hpp>
#include <vector>
using namespace std;



template <typename T>
class CUDADispatcher {
public:
  CUDADispatcher(): device(0) {};
  
  // Instances of parameters structures
  ParallelizationParameters par_parameters;
  AlgorithmParameters alg_parameters;

  // Initializes the device
  void initDevice()
  {
    // Get number of devices
    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0)
      {
	throw("No CUDA devices, quitting...");
      }

    // Select specific device - TODO: allow user to explicitly select it or choose the most powerful one
    cutilSafeCall(cudaSetDevice(device));
    cudaDeviceProp deviceProp;
    cutilSafeCall(cudaGetDeviceProperties(&deviceProp, device));

#ifdef CUDA_PINNED_MEMORY
    //enable (if supported) pinned memory
#if CUDART_VERSION >= 2020
    if(!deviceProp.canMapHostMemory) {
      fprintf(stderr, "Device %d does not support mapping CPU host memory!\n", device);
      exit(1);
    }
#if CUDART_VERSION >= 4000
    cutilSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost|cudaDeviceScheduleBlockingSync));
    //cutilSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleSpin));
#else //CUDART_VERSION >= 4000
    cutilSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif //CUDART_VERSION >= 4000
  
#else //CUDART_VERSION >= 2020
    fprintf(stderr, "CUDART version %d.%d does not support <cudaDeviceProp.canMapHostMemory> field\n", , CUDART_VERSION/1000, (CUDART_VERSION%100)/10);
    cutilDeviceReset(); 
    exit(1);
#endif //CUDART_VERSION >= 2020
#endif //CUDA_PINNED_MEMORY
  }

  // Run kernels on device with the specified input arguments
  void run()
  {
    int neighbours = par_parameters.thread_grain * par_parameters.num_threads_per_block * 4 /* num_neighbours */ *  sizeof(CUDAPixel) /* aligned to 4 */;
    int reduction = par_parameters.num_threads_per_block * sizeof(int); //max residual within each block
    int smemSize = neighbours + reduction; //working memory size of a block

    /* ???
    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (par_parameters.num_threads_per_block <= 32) ?  2 * smemElems * sizeof(CUDAPixel) : smemElems * sizeof(CUDAPixel);
    float * powt = NULL;
    if (alg_parameters.powalpha!=NULL) powt = alg_parameters.powalpha->get_table_device_addr();
    */

    //restore
    SPDkernel_wrapper(device_noisy_vect, device_residual_diffs, device_residuals, device_image_data, image_height, image_width, smemSize, alg_parameters, par_parameters);
    //backup
    backupKernel_wrapper(device_noisy_vect, device_image_data, image_height, image_width, smemSize, alg_parameters, par_parameters);

#if CUDART_VERSION >= 4000
    cudaDeviceSynchronize();
#else
    cudaThreadSynchronize();
#endif
  }


	
  // Get max residual
  residual_t MaxResidual()
  {
#ifndef CUDA_PINNED_MEMORY
    cudaMemcpy(residual_array, device_residuals, par_parameters.num_blocks * sizeof(residual_t), cudaMemcpyDeviceToHost);
#endif
    residual_t max_residual = 0;
    for(int i=0; i<par_parameters.num_blocks; i++)
      max_residual = _MAX(max_residual, residual_array[i]);
    return max_residual;
  }



  // Get average residual
  residual_t AvgResidual()
  {
#ifndef CUDA_PINNED_MEMORY
    cudaMemcpy(residual_array, device_residuals, par_parameters.num_blocks * sizeof(residual_t), cudaMemcpyDeviceToHost);
#endif
    residual_t total_residual = 0;
    for(int i=0; i<par_parameters.num_blocks; i++)
      total_residual += residual_array[i];
    return (total_residual / par_parameters.n_noisy);
  }



  void releaseMemory()
  {
    delete cuda_image;
#ifdef CUDA_PINNED_MEMORY
    cutilSafeCall(cudaFreeHost(host_noisy_vect));
    cutilSafeCall(cudaFreeHost(residual_array));
#else
    cutilSafeCall(cudaFree(device_noisy_vect));
    cutilSafeCall(cudaFree(device_residuals));
    free(residual_array);
#endif
    cutilSafeCall(cudaFree(device_residual_diffs));
  }



  void initMemory(int n_noisy, int height, int width, int threads_per_block, int grain)
  {
    par_parameters.n_noisy = n_noisy;
    par_parameters.num_blocks = (n_noisy + (threads_per_block * grain - 1)) / (threads_per_block * grain); //round up
    par_parameters.thread_grain = grain;
    par_parameters.num_threads_per_block = threads_per_block;
    image_width = width;
    image_height = height;
#ifdef DEBUG
    cerr << "Noisy points " << n_noisy
	 << " Grain " << grain << " Threads per block " << threads_per_block
	 << " Blocks " << par_parameters.num_blocks << "\n";
#endif

    //bitmap
    cuda_image = new CUDAImage(height, width);
#ifdef CUDA_PINNED_MEMORY
    cutilSafeCall(cudaHostGetDevicePointer((void **)&device_image_data, (void *)cuda_image->getData(), device));
#else
    device_image_data = cuda_image->getData();
#endif

    //noisy
#ifdef CUDA_PINNED_MEMORY
    cutilSafeCall(cudaHostAlloc((void **)&host_noisy_vect, n_noisy*sizeof(CUDAPoint), cudaHostAllocWriteCombined|cudaHostAllocMapped));
    cutilSafeCall(cudaHostGetDevicePointer((void **)&device_noisy_vect, (void *)host_noisy_vect, device));
#else
    cutilSafeCall(cudaMalloc((void **)&device_noisy_vect, n_noisy*sizeof(CUDAPoint)));
#endif

    //residual
    assert(par_parameters.num_blocks!=0);
#ifdef CUDA_PINNED_MEMORY
    cutilSafeCall(cudaHostAlloc((void **)&residual_array, sizeof(residual_t) * par_parameters.num_blocks, cudaHostAllocMapped));
    cutilSafeCall(cudaHostGetDevicePointer((void **)&device_residuals, (void *)residual_array, device));
#else
    cutilSafeCall(cudaMalloc((void **)&device_residuals, sizeof(residual_t) * par_parameters.num_blocks));
    residual_array = (residual_t *)malloc(sizeof(residual_t) * par_parameters.num_blocks);
#endif

    //diff
    cutilSafeCall(cudaMalloc((void **)&device_residual_diffs, n_noisy*sizeof(int)));

    //init diff and residual
    initializeClusterResidualDiffsKernel_wrapper(device_residual_diffs, device_residuals, n_noisy, par_parameters);
  }



  //Copy device-data to host (and vice versa)
  void host2device(vector<noisy<T> > &noisy_set, int n_noisy, Bitmap<T> &bmp, int height, int width)
  {
    //bitmap
#ifdef CUDA_PINNED_MEMORY
    for(int i=0; i<height; i++) {
      for(int j=0; j<width; j++) {
	CUDAPixel& cuda_pixel = cuda_image->get(i,j); //destination
	cuda_pixel.grey = cuda_pixel.original = bmp.get(j, i); 
	cuda_pixel.noisy = bmp.get_noisy(j, i);
      }
    }
#else //CUDA_PINNED_MEMORY
    CUDAPixel *buf = (CUDAPixel *)malloc(height * width * sizeof(CUDAPixel));
    CUDAPixel *buf_iterator = buf;
    for(int i=0; i<height; i++) {
      for(int j=0; j<width; j++) {
	buf_iterator->grey = bmp.get(j, i);
	buf_iterator->original = buf_iterator->grey;
	buf_iterator->noisy = bmp.get_noisy(j, i);
	++buf_iterator;
      }
    }
    cudaMemcpy(device_image_data, buf, height * width * sizeof(CUDAPixel), cudaMemcpyHostToDevice);
    free(buf);
#endif //CUDA_PINNED_MEMORY

    //noisy
#ifdef CUDA_PINNED_MEMORY
    for(int i=0; i<n_noisy; i++) {
      host_noisy_vect[i].x = noisy_set[i].c;
      host_noisy_vect[i].y = noisy_set[i].r;
    }
#else
    CUDAPoint *buf_noisy = (CUDAPoint *)malloc(n_noisy * sizeof(CUDAPoint));
    for(int i=0; i<n_noisy; i++) {
      buf_noisy[i].x = noisy_set[i].c;
      buf_noisy[i].y = noisy_set[i].r;
    }
    cudaMemcpy(device_noisy_vect, buf_noisy, n_noisy * sizeof(CUDAPoint), cudaMemcpyHostToDevice);
    free(buf_noisy);
#endif
  }



  void device2host(Bitmap<T> &bmp, int height, int width)
  {
    //bitmap
#ifdef CUDA_PINNED_MEMORY
    for(int i=0; i<height; i++)
      for(int j=0; j<width; j++)
	bmp.set(j, i, (cuda_image->get(i,j)).grey);
#else
    CUDAPixel *buf = (CUDAPixel *)malloc(height * width * sizeof(CUDAPixel));
    cudaMemcpy(buf, device_image_data, height * width * sizeof(CUDAPixel), cudaMemcpyDeviceToHost);
    CUDAPixel *buf_iterator = buf;
    for(int i=0; i<height; i++)
      for(int j=0; j<width; j++)
	bmp.set(j, i, (buf_iterator++)->grey);
    free(buf);
#endif
  }





private:
  //device
  residual_t *device_residuals; //global residuals
  CUDAPoint *device_noisy_vect; //noisy array
  CUDAPixel *device_image_data; //image data
  int *device_residual_diffs; //diffs

  //host
  CUDAImage *cuda_image; //CUDA-format image
  residual_t *residual_array; //global residuals
#ifdef CUDA_PINNED_MEMORY
  CUDAPoint *host_noisy_vect; //noisy array
#endif

  int image_width, image_height; //image size

  int device;  // Current device
};
#endif
