# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

ADD_DEFINITIONS(-DFF_OPENCL)

include_directories( ${OPENCL_INCLUDE_DIRS} )


set( TESTS                               
dotProduct farm_oclMap farm_oclReduce oclMap oclMapReduce oclReduce
oclStencilReduceLoop pipeFarmMap device_selection device_selection2
mixedpipe mixedpipe2
)


#if (NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC"))
#set(TESTS ${TESTS} test_taskf test_multi_output2 test_parforpipereduce test_pipe test_map test_mdf)
#endif ()

#set( TESTS_W_OMP test_parfor_basic)
#  

#if  ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#set (TESTS ${TESTS} ${TESTS_W_OMP})
#endif ( )

foreach( t ${TESTS} )
    add_executable( ${t} ${t}.cpp)
endforeach( t )

foreach( t ${TESTS} )
    target_link_libraries( ${t} ${CMAKE_THREAD_LIBS_INIT} ${OPENCL_LIBRARIES} )
endforeach( t )

foreach( t ${TESTS} )
add_custom_command(TARGET ${t} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/ff_opencl_datatypes.cl
                       $<TARGET_FILE_DIR:dotProduct>
                 )
endforeach( t )

foreach( t ${TESTS} )
    add_test( ${t} ${CMAKE_CURRENT_BINARY_DIR}/${t} )
endforeach( t )




