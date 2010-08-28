include_directories( ${PROJECT_SOURCE_DIR} )
add_executable( matmul matmul.cpp )
add_executable( ff_matmul ff_matmul.cpp )

target_link_libraries ( matmul pthread )
target_link_libraries ( ff_matmul pthread )

set_target_properties ( matmul      PROPERTIES COMPILE_FLAGS "-O3" )
set_target_properties ( matmul      PROPERTIES COMPILE_FLAGS "-O3" )
