#include FF
#include_directories (../../../)
include_directories (
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/spd
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/gaussian
  )

# This is a patch to cope with strange behaviour of cmake in 
# generating nvcc compilation string
# Here really -std=c++11 is needed. With make & cuda6.5 it works,
# but a for an unknown reason with cmake it doesn't work
set(CUDA_NVCC_FLAGS -use_fast_math; -Xcompiler -DNOT_USE_CMAKE_CONFIG)


if(APPLE)
	#SET(CMAKE_CXX_FLAGS " -stdlib=libstdc++")
        set(CUDA_NVCC_FLAGS -ccbin /usr/bin/clang++ -Xcompiler -stdlib=libstdc++ -gencode arch=compute_30,code=sm_30)
elseif(APPLE)
   #ADD_DEFINITIONS(-UUSE_CMAKE_CONFIG)
endif(APPLE)

# END ----------------------------------------------------------------------------


set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)


find_package(OpenCV)
#find_package(Threads)


set (SRCS_FF mainFF.cpp parameters.cpp)
include_directories (./)
ADD_DEFINITIONS(
   -std=c++11
   # Other flags
   -DNO_DEFAULT_MAPPING -finline-functions
)



#FF version

add_executable (denoiser ${SRCS_FF})
target_link_libraries (denoiser ${CMAKE_THREAD_LIBS_INIT})
TARGET_LINK_LIBRARIES (denoiser ${OpenCV_LIBS} )

#CUDA version


FIND_PACKAGE(CUDA)

if((CUDA_FOUND) AND NOT (APPLE))
  
  INCLUDE(FindCUDA)
  
  set (DENOISER_CUDAAUTO_SRCS mainCUDAAUTO.cu parameters.cpp)
  CUDA_ADD_EXECUTABLE ( denoiser_cuda_auto ${DENOISER_CUDAAUTO_SRCS} )
  target_link_libraries ( denoiser_cuda_auto ${CMAKE_THREAD_LIBS_INIT} )
  target_link_libraries ( denoiser_cuda_auto ${OpenCV_LIBS} )
  
else((CUDA_FOUND) AND NOT (APPLE))
if(NOT (APPLE))
  message (WARNING "Cuda not found: can't configure CUDA-enabled denoiser")
endif(NOT (APPLE))
endif((CUDA_FOUND) AND NOT (APPLE))
