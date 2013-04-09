// Define kernel to run on device.
// TODO: directly use CUDAImage class... It's quite stressful to access pixes by image[i*width+j]
#include "cuda_kernel.h"
#include "cuda_definitions.h"
#include <math.h>
#include <cutil_inline.h>
#include "sm_11_atomic_functions.h"
//#include <math.h>

__global__ void SPDkernel(CUDAPoint* __restrict__ noisy_vect, int n_noisy, int *residual_diffs, residual_t *residuals,  CUDAPixel* __restrict__  image_data, int height, int width,  AlgorithmParameters algparams, ParallelizationParameters parparams)
{
  //set pointers to shared memory portions (some pointer arithmetic)
  extern __shared__ int shmem[]; // aligned at 4 bytes
  CUDAPixel *neighbour_pixels = (CUDAPixel *) &shmem[0];
  residual_t *reduction = (residual_t *)(neighbour_pixels + (parparams.thread_grain * parparams.num_threads_per_block * 4));
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  int n_active_threads = 0;
  float alpha = algparams.alpha;
  float beta = algparams.beta;
  int grain = parparams.thread_grain;

  residual_t thread_red = 0, tmpd;
  int tmp;
  for (int g=0; g<grain; g++) {
    if((i*grain+g) >= n_noisy)
      break;
    ++n_active_threads;
    CUDAPoint& point = noisy_vect[i*grain+g];
    CUDAPixel pixel = image_data[point.row * width + point.col];
    // Get the up-to-four closest neighbours
    int num_neighbours = 4*(threadIdx.x*grain + g);
	
    if(point.row > 0)		
      neighbour_pixels[num_neighbours++] = image_data[(point.row - 1)*width + point.col];
    if(point.row < height - 1)
      neighbour_pixels[num_neighbours++] = image_data[(point.row + 1)*width + point.col];
    if(point.col > 0)
      neighbour_pixels[num_neighbours++] = image_data[(point.row)*width + point.col - 1];
    if(point.col < width - 1)
      neighbour_pixels[num_neighbours++] = image_data[(point.row)*width + point.col + 1];
	
    // Run the restoration algorithm
    float S;
    float Fu; 
    float Fu_prec = 256.0;
    unsigned char Fu_min_u = 0;
    float beta_ = beta / 2;
    for(int u=0; u<256; ++u) {
      Fu = 0.0;
      S = 0.0;
      for(int h = 4*(threadIdx.x*grain+g) ; h<num_neighbours; ++h) {
	S += (2-neighbour_pixels[h].noisy) * 
	  //__powf(abs(u - neighbour_pixels[h].grey), alpha);
	  __powf(abs(u - neighbour_pixels[h].original), alpha);
      }
      Fu += (abs(u - pixel.grey) + (beta_) * S); // (S1 + S2));
      if(Fu < Fu_prec)
	Fu_min_u = u;
      Fu_prec = Fu;
    }
    pixel.grey = image_data[point.row*width+point.col].grey = Fu_min_u;
 
    // Residual
    tmp = abs(((int) pixel.grey) - ((int) pixel.original)); //current diff
    tmpd = (residual_t)abs(tmp - residual_diffs[i*grain+g]); //current pass residual
    residual_diffs[i*grain+g] = tmp; //update diff

    // reduce for local iterations (grainwise)
#ifndef AVG_TERMINATION
    if (tmpd > thread_red)
      thread_red = tmpd;
#else
    thread_red += tmpd;
#endif
  }

  // Parallel reduce in shared memory (blockwise), should be scalable
#ifndef AVG_TERMINATION //max
  reduction[threadIdx.x] = thread_red;
  __syncthreads();
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (threadIdx.x < s)
      if (reduction[threadIdx.x] < reduction[threadIdx.x + s])
	reduction[threadIdx.x] = reduction[threadIdx.x + s];
    __syncthreads();
  }
  //blockwise max residual
  if (threadIdx.x==0)
    residuals[blockIdx.x] = reduction[0];

#else //avg
  reduction[threadIdx.x] = thread_red;
  __syncthreads();
  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (threadIdx.x < s)
      //if (reduction[threadIdx.x] < reduction[threadIdx.x + s])
      reduction[threadIdx.x] += reduction[threadIdx.x + s];
    __syncthreads();
  }
  
  if (threadIdx.x==0)
    residuals[blockIdx.x] = reduction[0]; //total block residual
  
#endif

  /* Should be not very scalable
     residuals[blockIdx.x] = 0;
     __syncthreads();
     atomicMax(&residuals[blockIdx.x],tmpd);
  */
}



//Kernel for the backup of the noisy pixels
__global__ void backupKernel(CUDAPoint * noisy_vect, int n_noisy, CUDAPixel *image_data, int height, int width, AlgorithmParameters algparams, ParallelizationParameters parparams) {
  int grain = parparams.thread_grain;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int g=0; g<grain; g++) {
    if((i*grain+g) >= n_noisy)
      break;
    CUDAPoint& point = noisy_vect[i*grain+g];
    image_data[point.row * width + point.col].original = image_data[point.row * width + point.col].grey;
  }
}




// Kernel for the initialization of the cluster residual difference array
__global__ void initializeClusterResidualDiffsKernel(int *diffs, residual_t *residuals, int n_noisy, int grain) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  // zero residuals
  if (threadIdx.x == 0)
    residuals[blockIdx.x] = (residual_t)0;
  //zero diff residuals
  //if(i >= n_noisy) return;
  int k;
  for(int j=0; j<grain; j++) {
    k = i*grain+j;
    if(k < n_noisy)
      diffs[k] = 0;
  }
}

//SPDkernel (wrapper)
void SPDkernel_wrapper(CUDAPoint *noisy_vect, int *residual_diffs, residual_t *residuals, CUDAPixel *image_data, int height, int width, int smemSize, AlgorithmParameters algparams, ParallelizationParameters parparams) {
  SPDkernel<<<parparams.num_blocks,parparams.num_threads_per_block,smemSize>>>
    (noisy_vect, parparams.n_noisy, residual_diffs, residuals, image_data, height, width, algparams, parparams);
}

//backup noisy pixels (wrapper)
void backupKernel_wrapper(CUDAPoint *noisy_vect, CUDAPixel *image_data, int height, int width, int smemSize, AlgorithmParameters algparams, ParallelizationParameters parparams) {
  backupKernel<<<parparams.num_blocks,parparams.num_threads_per_block,smemSize>>>
    (noisy_vect, parparams.n_noisy, image_data, height, width, algparams, parparams);
}

//initialize residuals (wrapper)
void initializeClusterResidualDiffsKernel_wrapper(int *residual_diffs, residual_t *residuals, int n_noisy, ParallelizationParameters parparams) {
  initializeClusterResidualDiffsKernel<<<parparams.num_blocks, parparams.num_threads_per_block>>>
    (residual_diffs, residuals, n_noisy, parparams.thread_grain);
}
