# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

set(CUDA_NVCC_FLAGS -Xcompiler -std=c++11 -gencode arch=compute_30,code=sm_30)

ADD_DEFINITIONS() 


CUDA_ADD_EXECUTABLE(cudaMap cudaMap.cu)




