# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

ADD_DEFINITIONS(-DUSE_CMAKE_CONFIG) 


set( TESTS                               
                        test_graphsearch
perf_parfor                               test_lb_affinity
perf_parfor2                              
perf_test1                                test_masterworker
perf_test_alloc1                          
perf_test_alloc2                          test_mpmc
perf_test_alloc3                          test_multi_input
perf_test_noalloc                         test_multi_input2
simplest                                  test_multi_input3
sizeof                                    test_multi_input4
test1                                     test_multi_input5
test1b                                    test_multi_input6
test2                                     test_multi_input7
test3                                     test_multi_input8
test3_farm                                test_multi_masterworker
test3b                                    test_multi_output
test4                                     
test5                                     test_noinput_pipe
test6                                     test_ofarm
test7                                     test_ofarm2
test_MISD                                 test_parfor
test_accelerator+pinning                  test_parfor2
test_accelerator                          test_parfor_multireduce 
test_accelerator2                         test_parfor_multireduce2
test_accelerator3                         test_parfor_unbalanced
test_accelerator_farm+pipe                
test_accelerator_ofarm                    test_pipe+masterworker
test_accelerator_ofarm_multiple_freezing  
test_accelerator_pipe+farm                test_pool1
test_accelerator_pipe                     test_pool2
test_bmpmc                                test_pool3
test_dataflow                             test_scheduling
test_dataflow2                            #test_scheduling2
test_devicequery                          test_sendq
test_dotprod_parfor                       test_spinBarrier
test_dt                                   test_stopstartall
test_farm+pipe                            test_stopstartthreads
test_farm                                 test_stopstartthreads2
test_ffthread                             
test_torus                                test_freeze
test_uBuffer
)

if (NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC"))
set(TESTS ${TESTS} test_taskf test_multi_output2 test_parforpipereduce test_pipe test_map test_mdf)
endif ()

#set( TESTS_W_OMP test_parfor_basic)
#  

#if  ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#set (TESTS ${TESTS} ${TESTS_W_OMP})
#endif ( )

foreach( t ${TESTS} )
    add_executable( ${t} ${t}.cpp)
endforeach( t )

foreach( t ${TESTS} )
    target_link_libraries( ${t} ${CMAKE_THREAD_LIBS_INIT} )
endforeach( t )

foreach( t ${TESTS} )
    add_test( ${t} ${CMAKE_CURRENT_BINARY_DIR}/${t} )
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
        add_subdirectory( ocl )
      endif (  )	
endif ( NOT OPENCL_FOUND )

# Distributed

find_package(ZeroMQ)
if(NOT ZeroMQ_FOUND)
  message (WARNING " 0mq not found: cannot configure distributed mode")
else(NOT ZeroMQ_FOUND)
  add_subdirectory( d )
endif(NOT ZeroMQ_FOUND)


