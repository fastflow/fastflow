#include FF
#include_directories (../../../)
include_directories (
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/spd
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/gaussian
  )

if(APPLE)
	#SET(CMAKE_CXX_FLAGS " -stdlib=libstdc++")
        set(CUDA_NVCC_FLAGS -ccbin /usr/bin/clang++ -Xcompiler -stdlib=libstdc++ -gencode arch=compute_30,code=sm_30)
elseif(APPLE)
   ADD_DEFINITIONS(-DNOT_USE_CMAKE_CONFIG)
endif(APPLE)

# END ----------------------------------------------------------------------------


#set(CMAKE_BUILD_TYPE Release)
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
add_test( denoiser ${CMAKE_CURRENT_BINARY_DIR}/denoiser -n50 ${CMAKE_CURRENT_SOURCE_DIR}/../test/lena512.bmp )
add_test( denoiser_validation ${CMAKE_COMMAND} -E compare_files ${CMAKE_CURRENT_BINARY_DIR}/RESTORED_lena512.bmp ${CMAKE_CURRENT_SOURCE_DIR}/../test/RESTORED_lena512-50REF.bmp ) 

#CUDA version

FIND_PACKAGE(CUDA)

if((CUDA_FOUND) AND NOT (APPLE))
  
  INCLUDE(FindCUDA)
  set(CUDA_NVCC_FLAGS -use_fast_math -std=c++11)

  
  set (DENOISER_CUDAAUTO_SRCS mainCUDAAUTO.cu parameters.cpp)
  CUDA_ADD_EXECUTABLE ( denoiser_cuda_auto ${DENOISER_CUDAAUTO_SRCS} )
  target_link_libraries ( denoiser_cuda_auto ${CMAKE_THREAD_LIBS_INIT} )
  target_link_libraries ( denoiser_cuda_auto ${OpenCV_LIBS} )
  add_test( denoiser_cuda_auto ${CMAKE_CURRENT_BINARY_DIR}/denoiser_cuda_auto -n50 -F10  ${CMAKE_CURRENT_SOURCE_DIR}/../test/test.mp4 ) 
  add_test( denoiser_cuda_validation ${CMAKE_COMMAND} -E compare_files ${CMAKE_CURRENT_BINARY_DIR}/RESTORED_test.avi ${CMAKE_CURRENT_SOURCE_DIR}/../test/RESTORED_test-50REF.avi )
else((CUDA_FOUND) AND NOT (APPLE))
if(NOT (APPLE))
  message (WARNING "Cuda not found: can't configure CUDA-enabled denoiser")
endif(NOT (APPLE))
endif((CUDA_FOUND) AND NOT (APPLE))
