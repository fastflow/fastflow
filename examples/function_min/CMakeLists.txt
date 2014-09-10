set ( FUNCMIN_SRCS funcmin.cpp )

set( TESTS funcmin )

include_directories (
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atomic
    ${CURRENT_SOURCE_DIR}
    )
add_executable ( funcmin ${FUNCMIN_SRCS} )
target_link_libraries ( funcmin   ${CMAKE_THREAD_LIBS_INIT} )

#set_target_properties ( funcmin PROPERTIES COMPILE_FLAGS "-DTRACE_FASTFLOW" )

add_test( funcmin ${CMAKE_CURRENT_BINARY_DIR}/funcmin 20 10000 5000 -1.5 1.5 50 0.000001 24 12 48 )
