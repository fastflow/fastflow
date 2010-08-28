set ( NQFF_SRCS nq_ff.cpp )
set ( NQFFACC_SRCS nq_ff_acc.cpp )
set ( NQ_SRCS nq.c )

include_directories (
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atomic
    ${CURRENT_SOURCE_DIR}
    )
add_executable ( nq ${NQ_SRCS} )
add_executable ( nq_ff ${NQFF_SRCS} )
add_executable ( nq_ff_acc ${NQFFACC_SRCS} )

target_link_libraries ( nq pthread )
target_link_libraries ( nq_ff pthread )
target_link_libraries ( nq_ff_acc pthread )

set_target_properties ( nq          PROPERTIES COMPILE_FLAGS "-O3 -Wall -DTRACE_FASTFLOW" )
set_target_properties ( nq_ff       PROPERTIES COMPILE_FLAGS "-O3 -Wall -DTRACE_FASTFLOW" )
set_target_properties ( nq_ff_acc   PROPERTIES COMPILE_FLAGS "-O3 -Wall -DTRACE_FASTFLOW" )
