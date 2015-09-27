# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

ADD_DEFINITIONS() 

include_directories( ${OPENCL_INCLUDE_DIRS} )

CUDA_ADD_EXECUTABLE(cudaMap cudaMap.cu)




