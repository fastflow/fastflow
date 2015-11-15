# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

set( TESTS                               
cudaMap cudaMapAccelerator farm_cudaMap farm_cudaStencil farm_cudaReduce dotProductCUDA test_pool1CUDA test_pool2CUDA cudaMapManaged
)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
   set(CUDA_NVCC_FLAGS -DCHECK) #-std=c++11 -gencode arch=compute_30,code=sm_30
else (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
   set(CUDA_NVCC_FLAGS -DCHECK -std=c++11) 
endif   (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")

foreach( t ${TESTS} )
    CUDA_ADD_EXECUTABLE( ${t} ${t}.cu)
    target_link_libraries( ${t} ${CMAKE_THREAD_LIBS_INIT} )
endforeach( t )

foreach( t ${TESTS} )
    add_test( ${t} ${CMAKE_CURRENT_BINARY_DIR}/${t} )
endforeach( t )
