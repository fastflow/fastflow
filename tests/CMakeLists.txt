# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

ADD_DEFINITIONS(-DUSE_CMAKE_CONFIG)

set( TESTS                                test_ffthread
perf_parfor                               test_freeze
perf_parfor2                              test_graphsearch
perf_test1                                test_lb_affinity
perf_test_alloc1                          test_map
perf_test_alloc2                          test_masterworker
perf_test_alloc3                          test_mpmc
perf_test_noalloc                         test_multi_input
test_multi_input2
simplest                                  test_multi_input3
sizeof                                    test_multi_input4
test1                                     test_multi_input5
test1b                                    test_multi_input6
test2                                     test_multi_input7
test3                                     test_multi_input8
test3_farm                                test_multi_masterworker
test3b                                    test_multi_output
test4                                     test_multi_output2
test5                                     test_noinput_pipe
test6                                     test_ofarm
test7                                     test_parfor
test_MISD                                 test_parfor2
test_accelerator+pinning                  test_parfor_multireduce
test_accelerator                          test_parfor_unbalanced
test_accelerator2                         test_parforpipereduce
test_accelerator3                         test_pipe+masterworker
test_accelerator_farm+pipe                test_pipe
test_accelerator_ofarm                    test_pool1
test_accelerator_ofarm_multiple_freezing  test_pool2
test_accelerator_pipe+farm                test_pool3
test_accelerator_pipe                     test_scheduling
test_bmpmc                             test_scheduling2
test_dataflow                             test_sendq
test_dataflow2                            test_spinBarrier
test_devicequery                          test_stopstartall
test_dotprod_parfor                       test_stopstartthreads
test_dt                                   test_stopstartthreads2
test_farm+pipe                            test_torus
test_farm                                 test_uBuffer)

#  

foreach( t ${TESTS} )
    add_executable( ${t} ${t}.cpp)
endforeach( t )

foreach( t ${TESTS} )
    target_link_libraries( ${t} ${CMAKE_THREAD_LIBS_INIT} )
endforeach( t )

add_test( test1 ${CMAKE_CURRENT_BINARY_DIR}/test1 3 1000 )
add_test( test2 ${CMAKE_CURRENT_BINARY_DIR}/test2 3 1000 )
add_test( test3 ${CMAKE_CURRENT_BINARY_DIR}/test3 1000 )
add_test( test3b ${CMAKE_CURRENT_BINARY_DIR}/test3b 0 1000 0 1 2 )
add_test( test4 ${CMAKE_CURRENT_BINARY_DIR}/test4 1000 3 )
add_test( test5 ${CMAKE_CURRENT_BINARY_DIR}/test5 1000 )
add_test( test6 ${CMAKE_CURRENT_BINARY_DIR}/test6 1000 )
add_test( test7 ${CMAKE_CURRENT_BINARY_DIR}/test7 500 )
add_test( test_accelerator ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator 3 1000 )
add_test( test_accelerator2 ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator2 3 1000 )
add_test( test_accelerator3 ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator3 3 1000 )
add_test( test_accelerator_farm+pipe ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator_farm+pipe 1000 3 3 )
add_test( perf_test1 ${CMAKE_CURRENT_BINARY_DIR}/perf_test1 3 100 1 10 1000 )
#add_test( perf_test2 ${CMAKE_CURRENT_BINARY_DIR}/perf_test2 3 100 1 10 1000 )
add_test( perf_test_alloc1 ${CMAKE_CURRENT_BINARY_DIR}/perf_test_alloc1 3 100 1 10 1000 )
add_test( perf_test_alloc2 ${CMAKE_CURRENT_BINARY_DIR}/perf_test_alloc2 3 100 1 10 1000 )
#add_test( perf_test_alloc3 ${CMAKE_CURRENT_BINARY_DIR}/perf_test_alloc3 1000000 80 2 ) # to be fixed in win
add_test( perf_test_noalloc ${CMAKE_CURRENT_BINARY_DIR}/perf_test_noalloc 1000000 1000 2 100000 )
add_test( test_freeze ${CMAKE_CURRENT_BINARY_DIR}/test_freeze 3 1000 10 )
add_test( test_dt ${CMAKE_CURRENT_BINARY_DIR}/test_dt 3 50 )
add_test( test_uBuffer ${CMAKE_CURRENT_BINARY_DIR}/test_uBuffer 1024 1000000 0 0)
add_test( test_scheduling ${CMAKE_CURRENT_BINARY_DIR}/test_scheduling 3 10 )
add_test( test_masterworker ${CMAKE_CURRENT_BINARY_DIR}/test_masterworker 3 1000 10 )
add_test( test_multi_masterworker ${CMAKE_CURRENT_BINARY_DIR}/test_multi_masterworker 3 1000 )
add_test( test_pipe+masterworker ${CMAKE_CURRENT_BINARY_DIR}/test_pipe+masterworker 2 3 1000 )
add_test( test_accelerator_pipe ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator_pipe 3 1000 )
add_test( test_accelerator_pipe+farm ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator_pipe+farm 1000 3 )
add_test( test_torus ${CMAKE_CURRENT_BINARY_DIR}/test_torus 4 1000 )
#if (NOT(GCC_TOO_OLD))
#  add_test( test_mpmc ${CMAKE_CURRENT_BINARY_DIR}/test_mpmc 10000 3 5 )
#endif (NOT(GCC_TOO_OLD))
add_test( test_map ${CMAKE_CURRENT_BINARY_DIR}/test_map 65536 3 )
#add_test( test_map2 ${CMAKE_CURRENT_BINARY_DIR}/test_map2 65536 3 10 )
add_test( test_spinBarrier ${CMAKE_CURRENT_BINARY_DIR}/test_spinBarrier 3 100)
add_test( test_multi_input ${CMAKE_CURRENT_BINARY_DIR}/test_multi_input 3 10000 )
#add_test( test_mapreduce ${CMAKE_CURRENT_BINARY_DIR}/test_mapreduce 1000000 3 )
#add_test( test_dotprod ${CMAKE_CURRENT_BINARY_DIR}/test_dotprod 10485760 3 )
add_test( test_ofarm ${CMAKE_CURRENT_BINARY_DIR}/test_ofarm 3 1000 )
add_test( test_accelerator_ofarm ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator_ofarm 3 1000 )
add_test( test_accelerator_ofarm_multiple_freezing ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator_ofarm_multiple_freezing 3 1000 )
add_test( test_parfor ${CMAKE_CURRENT_BINARY_DIR}/test_parfor )
add_test( test_parfor2 ${CMAKE_CURRENT_BINARY_DIR}/test_parfor )
add_test( test_dotprod_parfor ${CMAKE_CURRENT_BINARY_DIR}/test_dotprod_parfor )

#layer2 tests

add_subdirectory( layer2-tests-HAL )

# Distributed

find_package(ZeroMQ)
if(NOT ZeroMQ_FOUND)
  message (WARNING " 0mq not found: cannot configure distributed mode")
else(NOT ZeroMQ_FOUND)
  add_subdirectory( d )
endif(NOT ZeroMQ_FOUND)


