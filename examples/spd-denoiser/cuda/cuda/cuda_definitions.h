// Contains type, variable and function definitions for the CUDA implementation

#ifndef SPD_CUDA_DEFINITIONS
#define SPD_CUDA_DEFINITIONS

//#include "pow_table_cuda.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_utils/cutil.h>
#include <cuda_utils/cutil_inline_runtime.h>
#include <sstream>
#include <iostream>


using namespace std;

//#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )
#define IsPowerofTwo(A) ((A != 0) && ((A & (A - 1)) == 0))

// Structure for parallelization parameters
struct ParallelizationParameters
{
  int n_noisy;
  int num_blocks; 	// Number of thread blocks
  int num_threads_per_block; 	// Number of threads per block
  int thread_grain;
};
 
// Structure for algorithm-specific parameters
struct AlgorithmParameters
{
  // Parameters of the objective function
  float alpha;
  //pow_table_cuda *powalpha;
  float beta;
};



// Structure representing a single pixel
struct __align__(4) CUDAPixel {
  // Current greyscale value of the pixel
  unsigned char grey;
  // Original value of the pixel
  unsigned char original;
  // Noisy pixel flag
  bool noisy;

  // Default constructor
 CUDAPixel() : grey(0), original(0), noisy(false) {}
  
  // Constructor with explicit attribute settings
 CUDAPixel(unsigned char grey_, unsigned char original_, bool noisy_) :
  grey(grey_), original(original_), noisy(noisy_) {}

  // copy constructor
  __device__ CUDAPixel (const CUDAPixel &copy) :
  grey(copy.grey), original(copy.original), noisy(copy.noisy) {}
};



// Structure representing a pair of (x,y) coordinates; they are used to define clusters for the CUDA device
struct __align__(4) CUDAPoint {
  union
  {
    short int x;
    short int col;
  };
  
  union
  {
    short int y;
    short int row;
  };

  //default constructor
  __device__ CUDAPoint() : x(0), y(0) {}
  
  //explicit constructor
 CUDAPoint(short int x_, short int y_) : x(x_), y(y_) {}
};
		


// Class for the representation of an image with CUDAPixels
class CUDAImage
{
 private:
  // Pointer to actual data
  CUDAPixel *data;
  // Size of the image
  int height, width;
  
 public:
  
  // Get size of the image
  inline int getHeight() { return height; }
  inline int getWidth() { return width; }
  
  // Constructor, allocates data according to the size of the image
  CUDAImage(int height_, int width_)
    {
      // Set size
      height = height_;
      width = width_;
      int size = sizeof(CUDAPixel)*height*width;
#ifdef CUDA_PINNED_MEMORY
      cutilSafeCall(cudaHostAlloc((void **)&data, size, cudaHostAllocMapped)); //data: host (pinned)
#else
      cutilSafeCall(cudaMalloc((void **)&data, size)); //data: device
#endif
    }

#ifdef CUDA_PINNED_MEMORY
  // Get specified pixel
  CUDAPixel& get(int row, int col)
    {
      return data[row*width + col];
    }
#endif
  
  // Get data pointer
  inline CUDAPixel* getData() { return data; }
	
  // Destructor, frees memory
  ~CUDAImage() {
#ifdef CUDA_PINNED_MEMORY
    cudaFreeHost(data);
#else
    cudaFree(data);
#endif
  }
};

#endif
