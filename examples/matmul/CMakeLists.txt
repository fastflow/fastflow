include_directories( ${PROJECT_SOURCE_DIR} )
add_executable( matmul matmul.cpp )
add_executable( ff_matmul ff_matmul.cpp )
add_executable( ff_matmul_v2 ff_matmul_v2.cpp )

set( TESTS ff_matmul ff_matmul_v2 )

target_link_libraries ( matmul pthread )
target_link_libraries ( ff_matmul pthread )
target_link_libraries ( ff_matmul_v2 pthread )

set_target_properties ( matmul         PROPERTIES COMPILE_FLAGS "-O3" )
set_target_properties ( ff_matmul      PROPERTIES COMPILE_FLAGS "-O3" )
set_target_properties ( ff_matmul_v2   PROPERTIES COMPILE_FLAGS "-O3" )

add_test( ff_matmul    ${CMAKE_CURRENT_BINARY_DIR}/ff_matmul 3 1 )
add_test( ff_matmul_v2 ${CMAKE_CURRENT_BINARY_DIR}/ff_matmul 4 1 )