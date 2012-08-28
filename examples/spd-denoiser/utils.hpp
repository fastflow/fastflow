#ifndef _SPS_UTILS_HPP_
#define _SPS_UTILS_HPP_
//#include "utils.h"
#include <string>
#include <ff/platforms/platform.h>
#ifdef FF_WITH_CUDA
#include <cuda_runtime_api.h>
#include <cuda_utils/cutil.h>
#include <cuda_utils/cutil_inline_runtime.h>
#endif
using namespace std;

//extract the filename from a path
static string get_fname(string &path) {
  size_t p = path.find_last_of("/") + 1;
  return path.substr(p, path.length() - p);
}

//funzioni per le misure di tempo
static long int get_usec_from(long int s) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000 + tv.tv_usec) - s;
}

#ifdef FF_WITH_CUDA
static void get_cuda_env(int &corecount, bool verbose) {
  try
    {
      int deviceCount = 0;
      cudaGetDeviceCount(&deviceCount);
      cutilSafeCall(cudaGetDeviceCount(&deviceCount));
      if(verbose) {
	if (deviceCount == 0)
	  std::cerr << "There is no device supporting CUDA\n";
	else
	  std::cerr << "\nFound " << deviceCount << " CUDA Capable device(s)\n";
      }

      int dev, driverVersion = 0, runtimeVersion = 0;     
      for (dev = 0; dev < deviceCount; ++dev) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	if(verbose)
	  printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
#if CUDART_VERSION >= 2020
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	if (verbose)
	  printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
		 driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);
#endif
	if(verbose)
	  printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
		 deviceProp.major, deviceProp.minor);
	char msg[256];
	if(verbose) {
	  sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
		  (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
	  printf("%s",msg);
#if CUDART_VERSION >= 2000
	  printf("  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n", 
		 deviceProp.multiProcessorCount,
		 _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		 _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
#endif
	  printf("  GPU Clock Speed:                               %.2f GHz\n", deviceProp.clockRate * 1e-6f);
	}
	
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
#endif


#define _ABS(a)	   (((a) < 0) ? -(a) : (a))
#define _MAX(a, b) (((a) > (b)) ? (a) : (b))
#define _MIN(a, b) (((a) < (b)) ? (a) : (b))


#endif //_SPS_UTILS_HPP_
