set ( NQFF_SRCS nq_ff.cpp )
set ( NQFFACC_SRCS nq_ff_acc.cpp )
set ( NQ_SRCS nq.c )

set( TESTS nq_ff nq_ff_acc )

include_directories (
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atomic
    ${CURRENT_SOURCE_DIR}
    )
add_executable ( nq ${NQ_SRCS} )
add_executable ( nq_ff ${NQFF_SRCS} )
add_executable ( nq_ff_acc ${NQFFACC_SRCS} )


target_link_libraries ( nq  )
target_link_libraries ( nq_ff ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries ( nq_ff_acc ${CMAKE_THREAD_LIBS_INIT} )

set_target_properties ( nq          PROPERTIES COMPILE_FLAGS "-DTRACE_FASTFLOW" )
set_target_properties ( nq_ff       PROPERTIES COMPILE_FLAGS "-DTRACE_FASTFLOW" )
set_target_properties ( nq_ff_acc   PROPERTIES COMPILE_FLAGS "-DTRACE_FASTFLOW" )

add_test( nq_ff     ${CMAKE_CURRENT_BINARY_DIR}/nq_ff 16 4 4 1 )
add_test( nq_ff_acc ${CMAKE_CURRENT_BINARY_DIR}/nq_ff_acc 16 4 4 1 )
