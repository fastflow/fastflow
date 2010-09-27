set ( FIBFF_SRCS ff_fib.cpp )
set ( FIB_SRCS fib.cpp )

include_directories (
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atomic
    ${CURRENT_SOURCE_DIR}
    )
add_executable ( fib ${FIB_SRCS} )
add_executable ( ff_fib ${FIBFF_SRCS} )

target_link_libraries ( fib )
target_link_libraries ( ff_fib pthread )

set_target_properties ( fib          PROPERTIES COMPILE_FLAGS "-O3 -Wall " )
set_target_properties ( ff_fib       PROPERTIES COMPILE_FLAGS "-O3 -Wall " )

