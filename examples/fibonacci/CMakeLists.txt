set ( FIBFF_SRCS ff_fib.cpp )
set ( FIB_SRCS fib.cpp )

set( TESTS ff_fib )

include_directories (
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atomic
    ${CURRENT_SOURCE_DIR}
    )
add_executable ( fib ${FIB_SRCS} )
add_executable ( ff_fib ${FIBFF_SRCS} )

target_link_libraries ( fib )
target_link_libraries ( ff_fib ${CMAKE_THREAD_LIBS_INIT} )

#set_target_properties ( fib          PROPERTIES COMPILE_FLAGS "-O3 -Wall " )
#set_target_properties ( ff_fib       PROPERTIES COMPILE_FLAGS "-O3 -Wall " )

add_test( ff_fib    ${CMAKE_CURRENT_BINARY_DIR}/ff_fib 40 20 4 1 )
