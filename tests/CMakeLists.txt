# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

ADD_DEFINITIONS(-DUSE_CMAKE_CONFIG) 


set( TESTS                               
  latptr11                                  test_map
perf_parfor                               test_masterworker
perf_parfor2                              test_mdf
perf_test1                                
perf_test_alloc1                          
perf_test_alloc2                          test_multi_input
perf_test_alloc3                          test_multi_input2
perf_test_noalloc                         test_multi_input3
simplest                                  test_multi_input4
sizeof                                    test_multi_input5
test1                                     test_multi_input6
test1b                                    test_multi_input7
test2                                     test_multi_input8
test3                                     test_multi_masterworker
test3_farm                                test_multi_output
test3b                                    test_multi_output2
test4                                     test_nodeselector
test5                                     test_noinput_pipe
test6                                     test_ofarm
test7                                     test_ofarm2
test8                                     test_parfor
test_MISD                                 test_parfor2
test_accelerator+pinning                  test_parfor_multireduce
test_accelerator                          test_parfor_multireduce2
test_accelerator2                         test_parfor_unbalanced
test_accelerator3                         test_parforpipereduce
test_accelerator_farm+pipe                test_pipe+masterworker
test_accelerator_ofarm                    test_pipe
test_accelerator_ofarm_multiple_freezing  test_pool1
test_accelerator_pipe+farm                test_pool2
test_accelerator_pipe                     test_pool3
test_scheduling
test_dataflow                             
test_dataflow2                            test_sendq
test_devicequery                          test_spinBarrier
test_dotprod_parfor                       test_stopstartall
test_dt                                   test_stopstartthreads
test_eosw                                 test_stopstartthreads2
test_farm+pipe                            test_taskcallbacks
test_farm                                 test_taskf
test_ffthread                             test_torus
test_freeze                               test_torus2
test_graphsearch                          test_uBuffer
test_lb_affinity
test_blk                                  test_blk2
test_blk3                                 test_blk4
)

# # tests MPMC for x86 only
if ( (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR (
 CMAKE_SYSTEM_PROCESSOR MATCHES "i386") )
   add_subdirectory( mpmc )
endif ( )

#if  ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#set (TESTS ${TESTS} ${TESTS_W_OMP})
#endif ( )

foreach( t ${TESTS} )
    add_executable( ${t}_NONBLOCKING ${t}.cpp)
    add_executable(${t}_BLOCKING ${t}.cpp)
    set_target_properties (${t}_BLOCKING PROPERTIES COMPILE_DEFINITIONS "BLOCKING_MODE")
endforeach( t )

foreach( t ${TESTS} )
    target_link_libraries( ${t}_NONBLOCKING ${CMAKE_THREAD_LIBS_INIT} )
    target_link_libraries( ${t}_BLOCKING ${CMAKE_THREAD_LIBS_INIT} )
endforeach( t )

foreach( t ${TESTS} )
    add_test( ${t}_NONBLOCKING ${CMAKE_CURRENT_BINARY_DIR}/${t}_NONBLOCKING )
    add_test( ${t}_BLOCKING ${CMAKE_CURRENT_BINARY_DIR}/${t}_BLOCKING )
    set_tests_properties ( ${t}_NONBLOCKING PROPERTIES TIMEOUT 180)
    set_tests_properties ( ${t}_BLOCKING PROPERTIES TIMEOUT 180)
endforeach( t )

#test with special compilation parameters
add_executable( test_scheduling2 test_scheduling2.cpp)
set_target_properties(test_scheduling2 PROPERTIES COMPILE_DEFINITIONS LB_CALLBACK)
target_link_libraries( test_scheduling2 ${CMAKE_THREAD_LIBS_INIT} )
add_test( test_scheduling2 ${CMAKE_CURRENT_BINARY_DIR}/test_scheduling2 )


#layer2 tests

add_subdirectory( layer2-tests-HAL )

#OpenCL

find_package(OpenCL)
if ( NOT OPENCL_FOUND )
	message ( WARNING "OpenCL not found" )	
else  ( NOT OPENCL_FOUND )
      message (STATUS "OpenCL supported (version not detected) ")
      if (  ( CMAKE_SYSTEM_NAME MATCHES "Darwin" ) AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") )
        message ( WARNING "***** SKIPPING OPENCL tests -> GCC+opencl on OS/Yosemite require to patch system headers - See: http://hamelot.co.uk/programming/osx-gcc-dispatch_block_t-has-not-been-declared-invalid-typedef/" )
      else ( )
        MESSAGE(STATUS "Configuring >>tests/ocl<< directory")
        add_subdirectory( ocl )
      endif (  )	
endif ( NOT OPENCL_FOUND )

# CUDA

find_package(CUDA)
if (NOT CUDA_FOUND)
   message (WARING "CUDA not found")
else (NOT CUDA_FOUND)
   message (STATUS "CUDA supported, version" ${CUDA_VERSION_STRING})
   MESSAGE(STATUS "Configuring >>tests/cuda<< directory")
   add_subdirectory( cuda )
endif (NOT CUDA_FOUND)   

# Distributed

find_package(ZeroMQ)
if(NOT (ZMQ_FOUND))
  message (WARNING " 0mq not found: cannot configure distributed mode")
else(NOT (ZMQ_FOUND))
  add_subdirectory( d )
endif(NOT (ZMQ_FOUND))


