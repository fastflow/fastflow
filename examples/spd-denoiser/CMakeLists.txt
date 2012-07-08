set (SPD_SRCS driver.cpp utils.cpp parameters.cpp)
set (SPD_CUDA_SRCS ${SPD_SRCS} cuda/cuda_kernel.cu)

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
set (SPD_SRCS ${SPD_SRCS} XGetopt/XGetopt.cpp)
endif (CMAKE_SYSTEM_NAME MATCHES "Windows")

include_directories (
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/atomic
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_utils
  )

#if(COMMAND cmake_policy)
#		   cmake_policy(SET CMP0003 NEW)
#endif(COMMAND cmake_policy)

#add_executable (spd_ff_std_passes ${SPD_SRCS})
#add_executable (spd_ff_cluster_passes ${SPD_SRCS})
#add_executable (spd_ff_border_passes ${SPD_SRCS})
#add_executable (spd_ff_flat_passes ${SPD_SRCS})

add_executable (spd_ff ${SPD_SRCS})
#add_executable (spd_ff_cluster_time ${SPD_SRCS})
#add_executable (spd_ff_border_time ${SPD_SRCS})
#add_executable (spd_ff_flat_time ${SPD_SRCS})

#variants flags: -DCLUSTER -DBORDER -DFLAT -DALTERNATE -DCUDA
#output flags: -DTIME -DPASSES
#convergence flags: -DCONV_FIXE -DCONV_MINIMUM

#passes
#set_target_properties (spd_ff_std_passes PROPERTIES COMPILE_DEFINITIONS "PASSES;AVG_TERMINATION")
#set_target_properties (spd_ff_cluster_passes PROPERTIES COMPILE_DEFINITIONS "PASSES;CLUSTER;AVG_TERMINATION")
#set_target_properties (spd_ff_border_passes PROPERTIES COMPILE_DEFINITIONS "PASSES;BORDER;AVG_TERMINATION")
#set_target_properties (spd_ff_flat_passes PROPERTIES COMPILE_DEFINITIONS "PASSES;FLAT;AVG_TERMINATION")

#set_target_properties (spd_ff_std_passes PROPERTIES COMPILE_FLAGS "-Wall -O6")
#set_target_properties (spd_ff_cluster_passes PROPERTIES COMPILE_FLAGS "-Wall -O6")
#set_target_properties (spd_ff_border_passes PROPERTIES COMPILE_FLAGS "-Wall -O6")
#set_target_properties (spd_ff_flat_passes PROPERTIES COMPILE_FLAGS "-Wall -O6")
#target_link_libraries (spd_ff_std_passes ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries (spd_ff_cluster_passes ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries (spd_ff_border_passes ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries (spd_ff_flat_passes ${CMAKE_THREAD_LIBS_INIT})

#time
set_target_properties (spd_ff PROPERTIES COMPILE_DEFINITIONS "TIME;AVG_TERMINATION")
#set_target_properties (spd_ff_cluster_time PROPERTIES COMPILE_DEFINITIONS "TIME;CLUSTER;AVG_TERMINATION")
#set_target_properties (spd_ff_border_time PROPERTIES COMPILE_DEFINITIONS "TIME;BORDER;AVG_TERMINATION")
#set_target_properties (spd_ff_flat_time PROPERTIES COMPILE_DEFINITIONS "TIME;FLAT;AVG_TERMINATION")

#set_target_properties (spd_ff PROPERTIES COMPILE_FLAGS "-Wall -O6")
#set_target_properties (spd_ff_cluster_time PROPERTIES COMPILE_FLAGS "-Wall -O6")
#set_target_properties (spd_ff_border_time PROPERTIES COMPILE_FLAGS "-Wall -O6")
#set_target_properties (spd_ff_flat_time PROPERTIES COMPILE_FLAGS "-Wall -O6")
target_link_libraries (spd_ff ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries (spd_ff_cluster_time ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries (spd_ff_border_time ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries (spd_ff_flat_time ${CMAKE_THREAD_LIBS_INIT})

#tests
add_test( spd_ff ${CMAKE_CURRENT_BINARY_DIR}/spd_ff -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/lena/lena512_noise10.bmp )



#CUDA variants
#allowed flags: CUDA_PINNED_MEMORY
FIND_PACKAGE(CUDA)

if(CUDA_FOUND)
  LINK_DIRECTORIES("/usr/local/cuda/lib")
  LINK_DIRECTORIES("/usr/local/cuda/lib64")
  INCLUDE(FindCUDA)
  set(CUDA_NVCC_FLAGS "--use_fast_math")
  
  CUDA_ADD_EXECUTABLE ( spd_cuda ${SPD_CUDA_SRCS} )
  target_link_libraries ( spd_cuda "cudart" )
  set_target_properties ( spd_cuda PROPERTIES COMPILE_DEFINITIONS "CUDA;TIME;AVG_TERMINATION" )
  set_target_properties ( spd_cuda PROPERTIES COMPILE_FLAGS "-Wall -O6" )

  #CUDA_ADD_EXECUTABLE ( spd_cuda_passes ${SPD_CUDA_SRCS} )
  #target_link_libraries ( spd_cuda_passes "cudart" )
  #set_target_properties ( spd_cuda_passes PROPERTIES COMPILE_DEFINITIONS "CUDA;PASSES;AVG_TERMINATION" )
  #set_target_properties ( spd_cuda_passes PROPERTIES COMPILE_FLAGS "-Wall -O6" )

  add_test( spd_cuda ${CMAKE_CURRENT_BINARY_DIR}/spd_cuda -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/lena/lena512_noise10.bmp )

else(CUDA_FOUND)
  message (WARNING "Cuda not found: can't configure CUDA-enabled spd-denoiser")
endif(CUDA_FOUND)
