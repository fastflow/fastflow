# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

ADD_DEFINITIONS() #-DFF_OPENCL_LOG

include_directories( ${OPENCL_INCLUDE_DIRS} )


#set( TESTS                               
#helloKernel testRelease convolution2D oclStencilReduceLoop oclMap oclMap2 oclReduce oclMapReduce farm_oclMap farm_#oclReduce pipeFarmMap dotProduct device_selection dotProd_cpugpu device_selection2 mixedpipe mixedpipe2
#)

set(TESTS
	convolution2D
	device_selection
	device_selection2
	dotProd_cpugpu
	dotProduct
	dotProduct2
	farm_oclMap
	farm_oclReduce
	helloKernel
	helloKernel2
	mixedpipe
	mixedpipe2
	oclMap
	oclMap2
	oclMapReduce
	oclReduce
	oclStencilReduceLoop
	pipeFarmMap
	testRelease
)


foreach( t ${TESTS} )
    add_executable( ${t} ${t}.cpp)
endforeach( t )

foreach( t ${TESTS} )
    target_link_libraries( ${t} ${CMAKE_THREAD_LIBS_INIT} ${OPENCL_LIBRARIES} )
endforeach( t )

add_custom_command(TARGET dotProduct POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/ff_opencl_datatypes.cl
                       $<TARGET_FILE_DIR:dotProduct>
		       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/cl_code/dotProduct.cl
                       ${CMAKE_CURRENT_BINARY_DIR}/cl_code/dotProduct.cl
                 )

add_custom_command(TARGET oclMap2 POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/cl_code/oclMap.cl
                       ${CMAKE_CURRENT_BINARY_DIR}/cl_code/oclMap.cl
                 )

foreach( t ${TESTS} )
    add_test( ${t} ${CMAKE_CURRENT_BINARY_DIR}/${t} )
endforeach( t )


add_subdirectory( gameoflife )
add_subdirectory( helmholtz )



