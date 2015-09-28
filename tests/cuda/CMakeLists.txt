# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

set( TESTS                               
cudaMap cudaMapAccelerator farm_cudaMap farm_cudaStencil farm_cudaReduce dotProductCUDA test_pool1CUDA test_pool2CUDA cudaMapManaged
)

set(CUDA_NVCC_FLAGS -std=c++11 -Xcompiler -DCHECK) #-gencode arch=compute_30,code=sm_30

foreach( t ${TESTS} )
    CUDA_ADD_EXECUTABLE( ${t} ${t}.cu)
endforeach( t )

foreach( t ${TESTS} )
    add_test( ${t} ${CMAKE_CURRENT_BINARY_DIR}/${t} )
endforeach( t )