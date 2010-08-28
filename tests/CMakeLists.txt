include_directories(
    ${PROJECT_SOURCE_DIR}
    )
    
add_executable( test1 test1.cpp )
add_executable( test2 test2.cpp )
add_executable( test3 test3.cpp )
add_executable( test4 test4.cpp )
add_executable( test5 test5.cpp )
add_executable( test6 test6.cpp )
add_executable( test7 test7.cpp )
add_executable( perf_test1 perf_test1.cpp )
add_executable( perf_test2 perf_test2.cpp )
add_executable( test_accelerator test_accelerator.cpp )
add_executable( test_accelerator2 test_accelerator2.cpp )
add_executable( test_accelerator_farm+pipe test_accelerator_farm+pipe.cpp )
add_executable( test_freeze test_freeze.cpp )
add_executable( test_masterworker test_masterworker.cpp )
add_executable( test_scheduling test_scheduling.cpp )
add_executable( test_dt test_dt.cpp )
add_executable( perf_test_alloc1 perf_test_alloc1.cpp )
add_executable( perf_test_alloc2 perf_test_alloc2.cpp )
add_executable( perf_test_alloc3 perf_test_alloc3.cpp )
add_executable( test_uBuffer test_uBuffer.cpp )

set( TESTS test1 test2 test3 test4 test5 test6 test7
perf_test1 perf_test2 test_accelerator
test_accelerator2 test_accelerator_farm+pipe
test_freeze test_masterworker test_scheduling
test_dt perf_test_alloc1 perf_test_alloc2
perf_test_alloc3 test_uBuffer )

foreach( t ${TESTS} )
    target_link_libraries( ${t} pthread )
endforeach( t )

add_test( test1 ${CMAKE_CURRENT_BINARY_DIR}/test1 3 1000 )
add_test( test2 ${CMAKE_CURRENT_BINARY_DIR}/test2 3 1000 )
add_test( test3 ${CMAKE_CURRENT_BINARY_DIR}/test3 1000 )
add_test( test4 ${CMAKE_CURRENT_BINARY_DIR}/test4 1000 3 )
add_test( test5 ${CMAKE_CURRENT_BINARY_DIR}/test5 1000 )
add_test( test6 ${CMAKE_CURRENT_BINARY_DIR}/test6 1000 )
add_test( test7 ${CMAKE_CURRENT_BINARY_DIR}/test7 1000 )
add_test( test_accelerator ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator 3 1000 )
add_test( test_accelerator2 ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator2 3 1000 )
add_test( test_accelerator_farm+pipe ${CMAKE_CURRENT_BINARY_DIR}/test_accelerator_farm+pipe 1000 3 3 )
add_test( perf_test1 ${CMAKE_CURRENT_BINARY_DIR}/perf_test1 3 100 1 10 1000 )
add_test( perf_test2 ${CMAKE_CURRENT_BINARY_DIR}/perf_test2 3 100 1 10 1000 )
add_test( perf_test_alloc1 ${CMAKE_CURRENT_BINARY_DIR}/perf_test_alloc1 3 100 1 10 1000 )
add_test( perf_test_alloc2 ${CMAKE_CURRENT_BINARY_DIR}/perf_test_alloc2 3 100 1 10 1000 )
add_test( test_freeze ${CMAKE_CURRENT_BINARY_DIR}/test_freeze 3 1000 10 )
add_test( test_dt ${CMAKE_CURRENT_BINARY_DIR}/test_dt 3 50 )
add_test( test_uBuffer ${CMAKE_CURRENT_BINARY_DIR}/test_uBuffer 1024 3 )
add_test( test_scheduling ${CMAKE_CURRENT_BINARY_DIR}/test_scheduling 3 10 )
add_test( test_masterworker ${CMAKE_CURRENT_BINARY_DIR}/test_masterworker 3 1000 10 )
