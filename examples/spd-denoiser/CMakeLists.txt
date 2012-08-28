set (SPD_SRCS driver.cpp parameters.cpp)

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
set (SPD_SRCS ${SPD_SRCS} XGetopt/XGetopt.cpp)
endif (CMAKE_SYSTEM_NAME MATCHES "Windows")

set (SPD_CUDA_SRCS ${SPD_SRCS} cuda/cuda_kernel.cu)

include_directories (
  ${PROJECT_SOURCE_DIR}
  #${PROJECT_SOURCE_DIR}/atomic
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_utils
  )

add_executable (spd_ff ${SPD_SRCS}) #FLAT (default)
add_executable (spd_ff_cluster ${SPD_SRCS}) #CLUSTER
add_executable (spd_ff_nd ${SPD_SRCS}) #CONTINUOUS
add_executable (spd_ff_max ${SPD_SRCS}) #FLAT with PSNR-based termination
add_executable (spd_ff_time ${SPD_SRCS}) #FLAT with time misuration

#variants flags: -DCLUSTER -DBORDER -DFLAT -DALTERNATE -DFF_WITH_CUDA
#output flags: -DTIME -DPASSES
#convergence flags: -DCONV_FIXE -DCONV_MINIMUM

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra") #doesn't work in win
endif(CMAKE_COMPILER_IS_GNUCXX)

set_target_properties (spd_ff_cluster PROPERTIES COMPILE_DEFINITIONS "CLUSTER")
set_target_properties (spd_ff_nd PROPERTIES COMPILE_DEFINITIONS "CONTINUOUS")
set_target_properties (spd_ff_max PROPERTIES COMPILE_DEFINITIONS "MAX_TERMINATION")
set_target_properties (spd_ff_time PROPERTIES COMPILE_DEFINITIONS "TIME")

#set_target_properties (spd_ff PROPERTIES COMPILE_FLAGS "-O2")
#set_target_properties (spd_ff_cluster PROPERTIES COMPILE_FLAGS "-O2")
#set_target_properties (spd_ff_nd PROPERTIES COMPILE_FLAGS "-O2")
#set_target_properties (spd_ff_max PROPERTIES COMPILE_FLAGS "-O2")
#set_target_properties (spd_ff_time PROPERTIES COMPILE_FLAGS "-O2")

target_link_libraries (spd_ff ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (spd_ff_cluster ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (spd_ff_nd ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (spd_ff_max ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (spd_ff_time ${CMAKE_THREAD_LIBS_INIT})

#tests
add_test( spd_ff ${CMAKE_CURRENT_BINARY_DIR}/spd_ff -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/lena/lena512_noise10.bmp )
add_test( spd_ff_cluster ${CMAKE_CURRENT_BINARY_DIR}/spd_ff_cluster -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/lena/lena512_noise10.bmp )
add_test( spd_ff_nd ${CMAKE_CURRENT_BINARY_DIR}/spd_ff_nd -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/lena/lena512_noise10.bmp )
add_test( spd_ff_max ${CMAKE_CURRENT_BINARY_DIR}/spd_ff_max -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/lena/lena512_noise10.bmp )
add_test( spd_ff_time ${CMAKE_CURRENT_BINARY_DIR}/spd_ff_time -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/lena/lena512_noise10.bmp )



#CUDA variants
#allowed flags: FF_WITH_CUDA_PINNED_MEMORY
FIND_PACKAGE(CUDA)

if(CUDA_FOUND)
#  LINK_DIRECTORIES("/usr/local/cuda/lib")
#  LINK_DIRECTORIES("/usr/local/cuda/lib64")
  INCLUDE(FindCUDA)
  set(CUDA_NVCC_FLAGS "--use_fast_math")
  
  CUDA_ADD_EXECUTABLE ( spd_cuda ${SPD_CUDA_SRCS} )
  target_link_libraries ( spd_cuda )
  set_target_properties ( spd_cuda PROPERTIES COMPILE_DEFINITIONS "FF_WITH_CUDA" )
#  set_target_properties ( spd_cuda PROPERTIES COMPILE_FLAGS "-O2" )

  CUDA_ADD_EXECUTABLE ( spd_cuda_pinned ${SPD_CUDA_SRCS} )
  target_link_libraries ( spd_cuda_pinned )
  set_target_properties ( spd_cuda_pinned PROPERTIES COMPILE_DEFINITIONS "FF_WITH_CUDA;FF_WITH_CUDA_PINNED_MEMORY" )
#  set_target_properties ( spd_cuda_pinned PROPERTIES COMPILE_FLAGS "-O2" )

  #tests
  add_test( spd_cuda ${CMAKE_CURRENT_BINARY_DIR}/spd_cuda -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/lena/lena512_noise10.bmp )
  add_test( spd_cuda_pinned ${CMAKE_CURRENT_BINARY_DIR}/spd_cuda_pinned -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/lena/lena512_noise10.bmp )

else(CUDA_FOUND)
  message (WARNING "Cuda not found: can't configure CUDA-enabled spd-denoiser")
endif(CUDA_FOUND)
