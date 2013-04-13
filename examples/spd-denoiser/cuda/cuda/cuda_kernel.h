// Define kernel to run on device.
#include <cuda_runtime_api.h>
#include "cuda_definitions.h"
#include <convergence.hpp>
void SPDkernel_wrapper(CUDAPoint *noisy_vect, int *residual_diffs, residual_t *residuals, CUDAPixel *image_data, int height, int width, int smemSize, AlgorithmParameters algparams, ParallelizationParameters parparams);

void backupKernel_wrapper(CUDAPoint *, CUDAPixel *, int, int, int, AlgorithmParameters, ParallelizationParameters);

void initializeClusterResidualDiffsKernel_wrapper(int *residual_diffs, residual_t *residuals, int n_noisy, ParallelizationParameters parparams);

// Read pixel from image
__device__ CUDAPixel getPixel(CUDAPixel* image_data, int height, int width, size_t pitch, int row, int col);

// Set pixel in image
__device__ void setPixel(CUDAPixel* image_data, int height, int width, size_t pitch, int row, int col, unsigned char pixel);

// Kernel function:
// - clusters: array of pointers to clusters (which are arrays of CUDAPoint)
// - num_clusters: number of clusters (i.e. length of "clusters" array)
// - cluster_sizes: array of the sizes of each cluster (same length as "clusters")
// - cluster_diffs: array of residual differences for all clusters and for all points in the clusters
// - cluster_residuals: array of residuals for all clusters
// - image_data: array of pixels, row by row
// - height, width: of the image
// - image:pitch: provided by cudaMallocPitch
// - alpha, beta: regularization parameters
/*
__global__ void SPDkernel(CUDAPoint * __restrict__ noisy_vect, int n_noisy, int *residual_diffs, residual_t *residuals, CUDAPixel * __restrict__ image_data, int height, int width,  AlgorithmParameters algparams, ParallelizationParameters parparams);
*/

/*
// Kernel for the initialization of the cluster residual difference array
__global__ void initializeClusterResidualDiffsKernel(int *diffs, residual_t *residuals,int n_noisy, int grain);
*/

/*
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ T __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ T __smem[];
        return (T*)__smem;
    }
};
*/
