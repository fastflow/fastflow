# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

#ADD_DEFINITIONS(-DFF_OPENCL_LOG)

include_directories( ${OPENCL_INCLUDE_DIRS} )


set( TESTS                               
convolution2D oclStencilReduceLoop oclMap oclMap2 oclReduce oclMapReduce farm_oclMap farm_oclReduce pipeFarmMap dotProduct device_selection dotProd_cpugpu device_selection2 mixedpipe mixedpipe2
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

#foreach( t ${TESTS} )
add_custom_command(TARGET dotProduct POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/ff_opencl_datatypes.cl
                       $<TARGET_FILE_DIR:dotProduct>
                 )
#endforeach( t )

add_custom_command(TARGET oclMap2 POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/cl_code/oclMap.cl
                       ${CMAKE_CURRENT_BINARY_DIR}/cl_code/oclMap.cl
                 )

foreach( t ${TESTS} )
    add_test( ${t} ${CMAKE_CURRENT_BINARY_DIR}/${t} )
endforeach( t )


add_subdirectory( gameoflife )



